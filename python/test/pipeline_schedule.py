import struct
import cupy as cp
import networkx as nx
import os
import uuid

from mscclpp import (
    ProxyService,
    Transport,
)
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder


# KERNEL_FILE = "pipeline_kernel.cu"
# KERNEL_FILE = "pipeline_kernel_no_divergence.cu"
KERNEL_FILE = "pipeline_kernel_simplified_read.cu"


def connect_nvlink(group: mscclpp_comm.CommGroup, remote_nghrs: list):
    for n in remote_nghrs:
        assert type(n) is int
        assert 0 <= n < group.nranks
        assert n != group.my_rank

    tran = Transport.CudaIpc
    connections = group.make_connection(remote_nghrs, tran)
    return connections


class PipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[bid] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[bid] = sm send peers of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[bid] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[bid] = proxy send peers of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[bid] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[bid] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        node_types: dict,     # node_types[bid]: <0: reduce node; =0: root node; >0: broadcast node.
        nblocks,
        nthreads=1024,
    ):
        assert nblocks <= 128
        if nblocks > 100:
            print(f"Warning: nblocks={nblocks} > 100", flush=True)
        n_peers = max([len(recv_sm_channels.get(bid, [])) + len(recv_proxy_channels.get(bid, [])) for bid in range(nblocks)] +
                      [len(send_sm_channels.get(bid, [])) + len(send_proxy_channels.get(bid, [])) for bid in range(nblocks)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file=KERNEL_FILE,
            kernel_name="pipeline_schedule",
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nblocks = nblocks
        self.nthreads = nthreads
        self.data = data

        self.data_chunk_offsets = data_chunk_offsets
        self.data_chunk_sizes = data_chunk_sizes
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        block_recv_sm_ch_starts = []
        block_send_sm_ch_starts = []
        block_recv_proxy_ch_starts = []
        block_send_proxy_ch_starts = []
        recv_scratches_arr = []
        block_scratch_starts = []
        nrecvs_sm = []
        nsends_sm = []
        nrecvs_proxy = []
        nsends_proxy = []
        node_types_arr = []
        for bid in range(nblocks):
            assert (bid in recv_sm_channels or bid in send_sm_channels or 
                    bid in recv_proxy_channels or bid in send_proxy_channels)
            assert bid in data_chunk_offsets
            assert bid in data_chunk_sizes
            assert bid in node_types
            assert data_chunk_offsets[bid] + data_chunk_sizes[bid] <= total_chunks

            if node_types[bid] > 0:
                assert len(recv_sm_channels.get(bid, [])) + len(recv_proxy_channels.get(bid, [])) <= 1
            else:
                if bid in recv_sm_channels:
                    assert bid in recv_sm_scratches
                    len(recv_sm_scratches[bid]) == len(recv_sm_channels[bid])
                if bid in recv_proxy_channels:
                    assert bid in recv_proxy_scratches
                    len(recv_proxy_scratches[bid]) == len(recv_proxy_channels[bid])

            block_recv_sm_ch_starts.append(len(recv_sm_handles_arr))
            block_recv_proxy_ch_starts.append(len(recv_proxy_handles_arr))
            recv_sm_handles = [ch.device_handle().raw for ch in recv_sm_channels.get(bid, [])]
            recv_proxy_handles = [ch.device_handle().raw for ch in recv_proxy_channels.get(bid, [])]
            assert len(recv_sm_handles) + len(recv_proxy_handles) <= n_peers
            recv_sm_handles_arr += recv_sm_handles
            recv_proxy_handles_arr += recv_proxy_handles
            block_scratch_starts.append(len(recv_scratches_arr))
            if bid in recv_sm_scratches:
                assert len(recv_sm_scratches[bid]) == len(recv_sm_handles)
                assert all(scratch_buff.shape[0] == scratch_size for scratch_buff in recv_sm_scratches[bid])
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_sm_scratches[bid]]
            else:
                recv_scratches_arr += [struct.pack("P", 0)] * len(recv_sm_handles)
            if bid in recv_proxy_scratches:
                assert len(recv_proxy_scratches[bid]) == len(recv_proxy_handles)
                assert all(scratch_buff.shape[0] == scratch_size for scratch_buff in recv_proxy_scratches[bid])
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_proxy_scratches[bid]]
            else:
                recv_scratches_arr += [struct.pack("P", 0)] * len(recv_proxy_handles)
            nrecvs_sm.append(len(recv_sm_handles))
            nrecvs_proxy.append(len(recv_proxy_handles))

            block_send_sm_ch_starts.append(len(send_sm_handles_arr))
            block_send_proxy_ch_starts.append(len(send_proxy_handles_arr))
            send_sm_handles = [ch.device_handle().raw for ch in send_sm_channels.get(bid, [])]
            send_proxy_handles = [ch.device_handle().raw for ch in send_proxy_channels.get(bid, [])]
            assert len(send_sm_handles) + len(send_proxy_handles) <= n_peers
            send_sm_handles_arr += send_sm_handles
            send_proxy_handles_arr += send_proxy_handles
            nsends_sm.append(len(send_sm_handles))
            nsends_proxy.append(len(send_proxy_handles))
            node_types_arr.append(node_types[bid])

        recv_sm_handles_mem = cp.asarray(memoryview(b"".join(recv_sm_handles_arr)), dtype=cp.uint8)
        send_sm_handles_mem = cp.asarray(memoryview(b"".join(send_sm_handles_arr)), dtype=cp.uint8)
        recv_proxy_handles_mem = cp.asarray(memoryview(b"".join(recv_proxy_handles_arr)), dtype=cp.uint8)
        send_proxy_handles_mem = cp.asarray(memoryview(b"".join(send_proxy_handles_arr)), dtype=cp.uint8)
        recv_scratches_mem = cp.asarray(memoryview(b"".join(recv_scratches_arr)), dtype=cp.uint8)
        assert len(recv_sm_handles_arr) > 0 or recv_sm_handles_mem.data.ptr == 0
        assert len(send_sm_handles_arr) > 0 or send_sm_handles_mem.data.ptr == 0
        assert len(recv_proxy_handles_arr) > 0 or recv_proxy_handles_mem.data.ptr == 0
        assert len(send_proxy_handles_arr) > 0 or send_proxy_handles_mem.data.ptr == 0
        assert len(recv_scratches_arr) > 0 or recv_scratches_mem.data.ptr == 0
        block_recv_sm_ch_starts = cp.array(block_recv_sm_ch_starts, dtype=cp.int32)
        block_send_sm_ch_starts = cp.array(block_send_sm_ch_starts, dtype=cp.int32)
        block_recv_proxy_ch_starts = cp.array(block_recv_proxy_ch_starts, dtype=cp.int32)
        block_send_proxy_ch_starts = cp.array(block_send_proxy_ch_starts, dtype=cp.int32)
        block_scratch_starts = cp.array(block_scratch_starts, dtype=cp.int32)
        nrecvs_sm = cp.array(nrecvs_sm, dtype=cp.int32)
        nsends_sm = cp.array(nsends_sm, dtype=cp.int32)
        nrecvs_proxy = cp.array(nrecvs_proxy, dtype=cp.int32)
        nsends_proxy = cp.array(nsends_proxy, dtype=cp.int32)
        node_types_arr = cp.array(node_types_arr, dtype=cp.byte)

        assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(recv_scratches_arr) == n_recv_sm_channels + n_recv_proxy_channels
        assert block_recv_sm_ch_starts.shape[0] == nblocks and block_send_sm_ch_starts.shape[0] == nblocks
        assert block_recv_proxy_ch_starts.shape[0] == nblocks and block_send_proxy_ch_starts.shape[0] == nblocks
        assert block_scratch_starts.shape[0] == nblocks
        assert nrecvs_sm.shape[0] == nblocks and nsends_sm.shape[0] == nblocks
        assert nrecvs_proxy.shape[0] == nblocks and nsends_proxy.shape[0] == nblocks
        assert node_types_arr.shape[0] == nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_scratches_mem.data.ptr)
        self.params += struct.pack("P", block_recv_sm_ch_starts.data.ptr) + struct.pack("P", block_send_sm_ch_starts.data.ptr)
        self.params += struct.pack("P", block_recv_proxy_ch_starts.data.ptr) + struct.pack("P", block_send_proxy_ch_starts.data.ptr)
        self.params += struct.pack("P", block_scratch_starts.data.ptr)
        self.params += struct.pack("P", nrecvs_sm.data.ptr) + struct.pack("P", nsends_sm.data.ptr)
        self.params += struct.pack("P", nrecvs_proxy.data.ptr) + struct.pack("P", nsends_proxy.data.ptr)
        self.params += struct.pack("P", node_types_arr.data.ptr) + struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)

        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches, recv_scratches_mem,
                      block_recv_sm_ch_starts, block_send_sm_ch_starts,
                      block_recv_proxy_ch_starts, block_send_proxy_ch_starts,
                      block_scratch_starts,
                      nrecvs_sm, nsends_sm, nrecvs_proxy, nsends_proxy,
                      node_types_arr]
        self._data_starts_nelem_totals = {}
        self._params = {}
    

    def prepare_params(self, nelem_total, nelem_per_send):
        assert not self.use_schatch or nelem_per_send <= self.scratch_size
        assert nelem_total <= self.data.shape[0]
        assert nelem_per_send % 4 == 0  # aligned by int4

        if nelem_total in self._data_starts_nelem_totals:
            data_starts, nelem_totals = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.total_chunks == 0
            nelem_per_chunk = nelem_total // self.total_chunks

            assert all(self.data_chunk_offsets[bid] * nelem_per_chunk % 4 == 0 for bid in range(self.nblocks))  # aligned by int4
            data_starts = cp.array([self.data_chunk_offsets[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None):
        return self.get_func(nelem_total, nelem_per_send)(stream_ptr)


def verify_spanning_tree(G: nx.DiGraph, nranks: int, root: int):
    assert G.number_of_nodes() == nranks
    assert G.number_of_edges() == nranks - 1
    assert nx.is_weakly_connected(G)
    for v in G.nodes():
        assert G.in_degree(v) == (1 if v != root else 0)


IB_TRANSPORTS = {
    Transport.IB0,
    Transport.IB1,
    Transport.IB2,
    Transport.IB3,
    Transport.IB4,
    Transport.IB5,
    Transport.IB6,
    Transport.IB7,
}


def make_channels(group: mscclpp_comm.CommGroup, proxy_service: ProxyService, connections: dict,
                  connection_types: dict, recv_peers: list, send_peers: list, data: cp.ndarray,
                  scratch_size: int):
    recv_peers, send_peers = list(recv_peers), list(send_peers)
    recv_sm_channels, recv_proxy_channels = [], []
    recv_sm_scratches, recv_proxy_scratches = ([], []) if scratch_size is not None else (None, None)
    for dest in recv_peers:
        connect = connections[dest]
        recv_buf = cp.empty(scratch_size, dtype=cp.int32) if scratch_size is not None else data
        if connection_types[dest] == "sm":
            assert connect.transport() == Transport.CudaIpc
            if scratch_size is not None:
                recv_sm_scratches.append(recv_buf)
            recv_sm_channels.append(group.make_sm_channel(recv_buf, connect, dest))
        elif connection_types[dest] == "proxy":
            assert connect.transport() == Transport.CudaIpc or connect.transport() in IB_TRANSPORTS
            if scratch_size is not None:
                recv_proxy_scratches.append(recv_buf)
            recv_proxy_channels.append(
                group.make_proxy_channel(proxy_service, recv_buf, connect, dest))
        else:
            assert False
    for dest in send_peers:
        tran = connections[dest].transport()
        if connection_types[dest] == "sm":
            assert tran == Transport.CudaIpc
        elif connection_types[dest] == "proxy":
            assert tran == Transport.CudaIpc or tran in IB_TRANSPORTS
        else:
            assert False
    send_sm_channels = [group.make_sm_channel(data, connections[dest], dest) 
                        for dest in send_peers if connection_types[dest] == "sm"]
    send_proxy_channels = [group.make_proxy_channel(proxy_service, data, connections[dest], dest) 
                           for dest in send_peers if connection_types[dest] == "proxy"]
    if scratch_size is not None:
        return (recv_sm_channels, send_sm_channels,
                recv_proxy_channels, send_proxy_channels,
                recv_sm_scratches, recv_proxy_scratches)
    else:
        return (recv_sm_channels, send_sm_channels,
                recv_proxy_channels, send_proxy_channels)


def allreduce_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                     connections: dict, connection_types: dict, data: cp.ndarray,
                     scratch_size: int, proxy_service: ProxyService = None):
    for dest, connect in connections.items():
        transport = connect.transport()
        connect_type = connection_types[dest]
        assert connect_type in ["sm", "proxy"]
        assert (transport == Transport.CudaIpc or 
                (transport in IB_TRANSPORTS and
                 proxy_service is not None and
                 connect_type == "proxy"))

    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    assert all(cnt == k for cnt in chunk_counts.values())
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    recv_sm_scratches, recv_proxy_scratches = {}, {}
    node_types = {}
    data_chunk_offsets = {}
    data_chunk_sizes = {}
    nblocks = 0

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]

        children = list(test_G.successors(group.my_rank))

        if group.my_rank == u:
            # root
            tb_id = nblocks
            nblocks += 1
            (recv_sm_channels[tb_id], send_sm_channels[tb_id],
             recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
             recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id]) = \
                make_channels(group=group, proxy_service=proxy_service, connections=connections,
                              connection_types=connection_types, recv_peers=children,
                              send_peers=children, data=data, scratch_size=scratch_size)
            node_types[tb_id] = 0
            data_chunk_offsets[tb_id] = chunk
            data_chunk_sizes[tb_id] = Cs[u, i]
        else:
            # reduce node
            tb_id = nblocks
            nblocks += 1
            (recv_sm_channels[tb_id], send_sm_channels[tb_id],
             recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
             recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id]) = \
                make_channels(group=group, proxy_service=proxy_service, connections=connections,
                              connection_types=connection_types, recv_peers=children,
                              send_peers=list(test_G.predecessors(group.my_rank)),
                              data=data, scratch_size=scratch_size)
            assert len(send_sm_channels[tb_id]) + len(send_proxy_channels[tb_id]) <= 1
            node_types[tb_id] = -1
            data_chunk_offsets[tb_id] = chunk
            data_chunk_sizes[tb_id] = Cs[u, i]

            # broadcast node
            tb_id = nblocks
            nblocks += 1
            (recv_sm_channels[tb_id], send_sm_channels[tb_id],
             recv_proxy_channels[tb_id], send_proxy_channels[tb_id]) = \
                make_channels(group=group, proxy_service=proxy_service, connections=connections,
                              connection_types=connection_types,
                              recv_peers=list(test_G.predecessors(group.my_rank)),
                              send_peers=children, data=data, scratch_size=None)
            assert len(recv_sm_channels[tb_id]) + len(recv_proxy_channels[tb_id]) <= 1
            node_types[tb_id] = 1
            data_chunk_offsets[tb_id] = chunk
            data_chunk_sizes[tb_id] = Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                node_types=node_types, nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel


def allgather_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                     connections: dict, connection_types: dict, data: cp.ndarray,
                     proxy_service: ProxyService = None):
    for dest, connect in connections.items():
        transport = connect.transport()
        connect_type = connection_types[dest]
        assert connect_type in ["sm", "proxy"]
        assert (transport == Transport.CudaIpc or 
                (transport in IB_TRANSPORTS and
                 proxy_service is not None and
                 connect_type == "proxy"))

    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    assert all(cnt == k for cnt in chunk_counts.values())
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    node_types = {}
    data_chunk_offsets = {}
    data_chunk_sizes = {}
    nblocks = 0

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]

        children = list(test_G.successors(group.my_rank))

        # broadcast node
        tb_id = nblocks
        nblocks += 1
        (recv_sm_channels[tb_id], send_sm_channels[tb_id],
         recv_proxy_channels[tb_id], send_proxy_channels[tb_id]) = \
         make_channels(group=group, proxy_service=proxy_service, connections=connections,
                       connection_types=connection_types,
                       recv_peers=list(test_G.predecessors(group.my_rank)),
                       send_peers=children, data=data, scratch_size=None)
        assert len(recv_sm_channels[tb_id]) + len(recv_proxy_channels[tb_id]) <= 1
        node_types[tb_id] = 1
        data_chunk_offsets[tb_id] = chunk
        data_chunk_sizes[tb_id] = Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=0,
                recv_sm_scratches={}, recv_proxy_scratches={},
                node_types=node_types, nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel


def reduce_scatter_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                          connections: dict, connection_types: dict, data: cp.ndarray,
                          scratch_size: int, proxy_service: ProxyService = None):
    for dest, connect in connections.items():
        transport = connect.transport()
        connect_type = connection_types[dest]
        assert connect_type in ["sm", "proxy"]
        assert (transport == Transport.CudaIpc or 
                (transport in IB_TRANSPORTS and
                 proxy_service is not None and
                 connect_type == "proxy"))

    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    assert all(cnt == k for cnt in chunk_counts.values())
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    recv_sm_scratches, recv_proxy_scratches = {}, {}
    node_types = {}
    data_chunk_offsets = {}
    data_chunk_sizes = {}
    nblocks = 0

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]

        children = list(test_G.successors(group.my_rank))

        # reduce node
        tb_id = nblocks
        nblocks += 1
        (recv_sm_channels[tb_id], send_sm_channels[tb_id],
            recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
            recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id]) = \
            make_channels(group=group, proxy_service=proxy_service, connections=connections,
                            connection_types=connection_types, recv_peers=children,
                            send_peers=list(test_G.predecessors(group.my_rank)),
                            data=data, scratch_size=scratch_size)
        assert len(send_sm_channels[tb_id]) + len(send_proxy_channels[tb_id]) <= 1
        node_types[tb_id] = -1
        data_chunk_offsets[tb_id] = chunk
        data_chunk_sizes[tb_id] = Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                node_types=node_types, nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel
