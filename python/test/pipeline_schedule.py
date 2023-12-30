import struct
import cupy as cp
import networkx as nx
import os

from mscclpp import Transport
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder


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
        recv_channels: dict,  # recv_peers[bid] = recv peers of tree
        send_channels: dict,  # send_peers[bid] = send peers of tree
        data: cp.ndarray,
        data_offsets: dict,   # data_offsets[bid] = offset of tree
        data_sizes: dict,     # data_sizes[bid] = data size of tree
        scratch_size: int,
        recv_scratches: dict,
        node_types: dict,     # node_types[bid]: <0: reduce node; =0: root node; >0: broadcast node.
        nelem_per_send: int,
        nblocks,
        nthreads=1024,
    ):
        n_peers = max([len(l) for l in recv_channels.values()] +
                      [len(l) for l in send_channels.values()] + [1])
        n_recv_channels = max(1, sum(len(l) for l in recv_channels.values()))
        n_send_channels = max(1, sum(len(l) for l in send_channels.values()))
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file="pipeline_kernel.cu",
            kernel_name="pipeline_schedule",
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nblocks = nblocks
        self.nthreads = nthreads

        assert nelem_per_send > 0
        if any(t <= 0 for t in node_types.values()):
            assert nelem_per_send <= scratch_size

        recv_handles_arr = []
        send_handles_arr = []
        block_recv_ch_starts = []
        block_send_ch_starts = []
        recv_scratches_arr = []
        nrecvs = []
        nsends = []
        node_types_arr = []
        data_starts = []
        nelem_totals = []
        for bid in range(nblocks):
            assert bid in recv_channels or bid in send_channels
            assert bid in data_offsets
            assert bid in data_sizes
            assert bid in node_types

            assert data_offsets[bid] + data_sizes[bid] <= data.shape[0]

            if node_types[bid] > 0:
                assert bid not in recv_channels or len(recv_channels[bid]) <= 1
            elif bid in recv_channels:
                assert bid in recv_scratches
                assert len(recv_scratches[bid]) == len(recv_channels[bid])

            block_recv_ch_starts.append(len(recv_handles_arr))
            recv_handles = [ch.device_handle().raw for ch in recv_channels.get(bid, [])]
            assert len(recv_handles) <= n_peers
            recv_handles_arr += recv_handles
            if bid in recv_scratches:
                assert len(recv_scratches[bid]) == len(recv_handles)
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_scratches[bid]]
            else:
                recv_scratches_arr += [struct.pack("P", 0)] * len(recv_handles)
            nrecvs.append(len(recv_handles))

            block_send_ch_starts.append(len(send_handles_arr))
            send_handles = [ch.device_handle().raw for ch in send_channels.get(bid, [])]
            assert len(send_handles) <= n_peers
            send_handles_arr += send_handles
            nsends.append(len(send_handles))
            
            node_types_arr.append(node_types[bid])
            data_starts.append(data_offsets[bid])
            nelem_totals.append(data_sizes[bid])

        assert len(recv_handles_arr) > 0 or len(send_handles_arr) > 0
        if len(recv_handles_arr) == 0:
            recv_handles_arr.append(bytes(len(send_handles_arr[0])))
            recv_scratches_arr.append(struct.pack("P", 0))
        if len(send_handles_arr) == 0:
            send_handles_arr.append(bytes(len(recv_handles_arr[0])))
        
        recv_handles_mem = cp.asarray(memoryview(b"".join(recv_handles_arr)), dtype=cp.uint8)
        send_handles_mem = cp.asarray(memoryview(b"".join(send_handles_arr)), dtype=cp.uint8)
        recv_scratches_mem = cp.asarray(memoryview(b"".join(recv_scratches_arr)), dtype=cp.uint8)
        assert len(recv_handles_arr) > 0 or recv_handles_mem.data.ptr == 0
        assert len(send_handles_arr) > 0 or send_handles_mem.data.ptr == 0
        assert len(recv_scratches_arr) > 0 or recv_scratches_mem.data.ptr == 0
        block_recv_ch_starts = cp.array(block_recv_ch_starts, dtype=cp.int32)
        block_send_ch_starts = cp.array(block_send_ch_starts, dtype=cp.int32)
        nrecvs = cp.array(nrecvs, dtype=cp.int32)
        nsends = cp.array(nsends, dtype=cp.int32)
        node_types_arr = cp.array(node_types_arr, dtype=cp.byte)
        data_starts = cp.array(data_starts, dtype=cp.uint64)
        nelem_totals = cp.array(nelem_totals, dtype=cp.uint64)

        assert len(recv_handles_arr) == n_recv_channels and len(send_handles_arr) == n_send_channels
        assert len(recv_scratches_arr) == n_recv_channels
        assert block_recv_ch_starts.shape[0] == nblocks and block_send_ch_starts.shape[0] == nblocks
        assert nrecvs.shape[0] == nblocks and nsends.shape[0] == nblocks
        assert node_types_arr.shape[0] == nblocks
        assert data_starts.shape[0] == nblocks
        assert nelem_totals.shape[0] == nblocks

        self.params = b""
        self.params += struct.pack("P", recv_handles_mem.data.ptr) + struct.pack("P", send_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_scratches_mem.data.ptr)
        self.params += struct.pack("P", block_recv_ch_starts.data.ptr) + struct.pack("P", block_send_ch_starts.data.ptr)
        self.params += struct.pack("P", nrecvs.data.ptr) + struct.pack("P", nsends.data.ptr)
        self.params += struct.pack("P", node_types_arr.data.ptr)
        self.params += struct.pack("P", data_starts.data.ptr)
        self.params += struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr) + struct.pack("Q", scratch_size)
        self.params += struct.pack("P", data.data.ptr)

        # keep references to avoid garbage collection
        self._temp = [recv_channels, send_channels,
                      recv_handles_mem, send_handles_mem,
                      data, recv_scratches, recv_scratches_mem,
                      block_recv_ch_starts, block_send_ch_starts, nrecvs, nsends, 
                      node_types_arr, data_starts, nelem_totals]

    def __call__(self):
        return self._kernel.launch_kernel(self.params, self.nblocks, self.nthreads, 0, None)


def verify_spanning_tree(G: nx.DiGraph, nranks: int, root: int):
    assert G.number_of_nodes() == nranks
    assert G.number_of_edges() == nranks - 1
    assert nx.is_weakly_connected(G)
    for v in G.nodes():
        assert G.in_degree(v) == (1 if v != root else 0)


def allreduce_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                     connections: dict, data: cp.ndarray, allreduce_length: int,
                     nelem_per_send: int, scratch_size: int):
    assert allreduce_length % (k * group.nranks) == 0
    assert scratch_size >= nelem_per_send
    assert allreduce_length == data.shape[0]

    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks
    nelem_per_chunk = allreduce_length // total_chunks

    recv_scratches = {}
    recv_channels = {}
    send_channels = {}
    node_types = {}
    data_offsets = {}
    data_sizes = {}
    nblocks = 0

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]
        offset = nelem_per_chunk * chunk

        children = list(test_G.successors(group.my_rank))

        if group.my_rank == u:
            # root
            tb_id = nblocks
            nblocks += 1
            recv_scratches[tb_id] = [cp.zeros(scratch_size, dtype=cp.int32) for _ in children]
            recv_channels[tb_id] = [group.make_sm_channel(recv_scratches[tb_id][idx], connections[dest], dest) 
                                    for idx, dest in enumerate(children)]
            send_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                    for dest in children]
            node_types[tb_id] = 0
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]
        else:
            # reduce node
            tb_id = nblocks
            nblocks += 1
            recv_scratches[tb_id] = [cp.zeros(scratch_size, dtype=cp.int32) for _ in children]
            recv_channels[tb_id] = [group.make_sm_channel(recv_scratches[tb_id][idx], connections[dest], dest) 
                                    for idx, dest in enumerate(children)]
            send_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                    for dest in test_G.predecessors(group.my_rank)]
            assert len(send_channels[tb_id]) <= 1
            node_types[tb_id] = -1
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

            # broadcast node
            tb_id = nblocks
            nblocks += 1
            recv_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                    for dest in test_G.predecessors(group.my_rank)]
            send_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                    for dest in children]
            assert len(recv_channels[tb_id]) <= 1
            node_types[tb_id] = 1
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

    args = dict(recv_channels=recv_channels, send_channels=send_channels, data=data, 
                data_offsets=data_offsets, data_sizes=data_sizes, scratch_size=scratch_size, 
                recv_scratches=recv_scratches, node_types=node_types, nelem_per_send=nelem_per_send, 
                nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel


def allgather_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                     connections: dict, data: cp.ndarray, allgather_length: int,
                     nelem_per_send: int):
    assert allgather_length % (k * group.nranks) == 0
    assert allgather_length == data.shape[0]

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
    nelem_per_chunk = allgather_length // total_chunks

    recv_scratches = {}
    recv_channels = {}
    send_channels = {}
    node_types = {}
    data_offsets = {}
    data_sizes = {}
    nblocks = 0

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]
        offset = nelem_per_chunk * chunk

        children = list(test_G.successors(group.my_rank))

        # broadcast node
        tb_id = nblocks
        nblocks += 1
        recv_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                for dest in test_G.predecessors(group.my_rank)]
        send_channels[tb_id] = [group.make_sm_channel(data, connections[dest], dest) 
                                for dest in children]
        assert len(recv_channels[tb_id]) <= 1
        node_types[tb_id] = 1
        data_offsets[tb_id] = offset
        data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

    args = dict(recv_channels=recv_channels, send_channels=send_channels, data=data, 
                data_offsets=data_offsets, data_sizes=data_sizes, scratch_size=0, 
                recv_scratches=recv_scratches, node_types=node_types, nelem_per_send=nelem_per_send, 
                nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel
