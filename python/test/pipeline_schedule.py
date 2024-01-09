import struct
import cupy as cp
import networkx as nx
import os
import tempfile

from mscclpp import (
    ProxyService,
    Transport,
)
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder


KERNEL_FILE = "pipeline_kernel.cu"
# KERNEL_FILE = "pipeline_kernel_simplified.cu"


def connect_nvlink(group: mscclpp_comm.CommGroup, remote_nghrs: list):
    for n in remote_nghrs:
        assert type(n) is int
        assert 0 <= n < group.nranks
        assert n != group.my_rank

    tran = Transport.CudaIpc
    connections = group.make_connection(remote_nghrs, tran)
    return connections


class CupyKernelBuilder:
    kernel_map: dict = {}

    def get_key(self, kernel_name, macro_dict):
        return kernel_name + "-".join(f"{key}={macro_dict[key]}" for key in sorted(macro_dict))

    def __init__(self, file: str, kernel_name: str, file_dir: str = None, macro_dict: dict = {}):
        kernel_key = self.get_key(kernel_name, macro_dict)
        if kernel_key in self.kernel_map:
            self._kernel = self.kernel_map[kernel_key]
            return
        self._tempdir = tempfile.TemporaryDirectory(suffix=f"{os.getpid()}")
        self._current_file_dir = file_dir if file_dir else os.path.dirname(os.path.abspath(__file__))
        self.macros = None
        if file_dir:
            self.macros = ["-D{}={}".format(macro, value) for macro, value in macro_dict.items()]
        device = cp.cuda.Device()
        device_id = device.id
        ptx = self._compile_cuda(os.path.join(self._current_file_dir, file), f"{kernel_name}.ptx", device_id)
        self._kernel = Kernel(ptx, kernel_name, device_id)
        self.kernel_map[kernel_key] = self._kernel

    def _compile_cuda(self, source_file, output_file, device, std_version="c++17"):
        include_dir = os.path.join(self._current_file_dir, "../../include")
        compute_capa = device.compute_capability
        assert len(compute_capa) == 2
        major, minor = compute_capa
        cuda_home = os.environ.get("CUDA_HOME")
        nvcc = os.path.join(cuda_home, "bin/nvcc") if cuda_home else "nvcc"
        command = [
            nvcc,
            f"-std={std_version}",
            "-ptx",
            "-Xcompiler",
            "-Wall,-Wextra",
            f"-I{include_dir}",
            f"{source_file}",
            f"--gpu-architecture=compute_{major}{minor}",
            f"--gpu-code=sm_{major}{minor},compute_{major}{minor}",
            "-o",
            f"{self._tempdir.name}/{output_file}",
        ]
        if self.macros:
            command += self.macros
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, bufsize=1)
            return cp.RawModule(path=f"{self._tempdir.name}/{output_file}").get_function("pipeline_schedule")
        except subprocess.CalledProcessError as e:
            print(e.stderr, end="")
            raise RuntimeError("Compilation failed: ", " ".join(command))

    def get_compiled_kernel(self):
        return self._kernel

    def __del__(self):
        if hasattr(self, "_tempdir"):
            self._tempdir.cleanup()


class PipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[bid] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[bid] = sm send peers of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[bid] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[bid] = proxy send peers of tree
        data: cp.ndarray,
        data_offsets: dict,   # data_offsets[bid] = offset of tree
        data_sizes: dict,     # data_sizes[bid] = data size of tree
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        node_types: dict,     # node_types[bid]: <0: reduce node; =0: root node; >0: broadcast node.
        nelem_per_send: int,
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

        assert nelem_per_send > 0
        if any(t <= 0 for t in node_types.values()):
            assert nelem_per_send <= scratch_size

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
        data_starts = []
        nelem_totals = []
        for bid in range(nblocks):
            assert (bid in recv_sm_channels or bid in send_sm_channels or 
                    bid in recv_proxy_channels or bid in send_proxy_channels)
            assert bid in data_offsets
            assert bid in data_sizes
            assert bid in node_types

            assert data_offsets[bid] + data_sizes[bid] <= data.shape[0]

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
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_sm_scratches[bid]]
            else:
                recv_scratches_arr += [struct.pack("P", 0)] * len(recv_sm_handles)
            if bid in recv_proxy_scratches:
                assert len(recv_proxy_scratches[bid]) == len(recv_proxy_handles)
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
            data_starts.append(data_offsets[bid])
            nelem_totals.append(data_sizes[bid])

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
        data_starts = cp.array(data_starts, dtype=cp.uint64)
        nelem_totals = cp.array(nelem_totals, dtype=cp.uint64)

        assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(recv_scratches_arr) == n_recv_sm_channels + n_recv_proxy_channels
        assert block_recv_sm_ch_starts.shape[0] == nblocks and block_send_sm_ch_starts.shape[0] == nblocks
        assert block_recv_proxy_ch_starts.shape[0] == nblocks and block_send_proxy_ch_starts.shape[0] == nblocks
        assert block_scratch_starts.shape[0] == nblocks
        assert nrecvs_sm.shape[0] == nblocks and nsends_sm.shape[0] == nblocks
        assert nrecvs_proxy.shape[0] == nblocks and nsends_proxy.shape[0] == nblocks
        assert node_types_arr.shape[0] == nblocks
        assert data_starts.shape[0] == nblocks
        assert nelem_totals.shape[0] == nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_scratches_mem.data.ptr)
        self.params += struct.pack("P", block_recv_sm_ch_starts.data.ptr) + struct.pack("P", block_send_sm_ch_starts.data.ptr)
        self.params += struct.pack("P", block_recv_proxy_ch_starts.data.ptr) + struct.pack("P", block_send_proxy_ch_starts.data.ptr)
        self.params += struct.pack("P", block_scratch_starts.data.ptr)
        self.params += struct.pack("P", nrecvs_sm.data.ptr) + struct.pack("P", nsends_sm.data.ptr)
        self.params += struct.pack("P", nrecvs_proxy.data.ptr) + struct.pack("P", nsends_proxy.data.ptr)
        self.params += struct.pack("P", node_types_arr.data.ptr)
        self.params += struct.pack("P", data_starts.data.ptr)
        self.params += struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr) + struct.pack("Q", scratch_size)
        self.params += struct.pack("P", data.data.ptr)

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
                      node_types_arr, data_starts, nelem_totals]

    def __call__(self):
        return self._kernel.launch_kernel(self.params, self.nblocks, self.nthreads, 0, None)


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
        recv_buf = cp.zeros(scratch_size, dtype=cp.int32) if scratch_size is not None else data
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
                     allreduce_length: int, nelem_per_send: int, scratch_size: int,
                     proxy_service: ProxyService = None):
    assert allreduce_length % (k * group.nranks) == 0
    assert scratch_size >= nelem_per_send
    assert allreduce_length == data.shape[0]
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
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks
    nelem_per_chunk = allreduce_length // total_chunks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    recv_sm_scratches, recv_proxy_scratches = {}, {}
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
            (recv_sm_channels[tb_id], send_sm_channels[tb_id],
             recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
             recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id]) = \
                make_channels(group=group, proxy_service=proxy_service, connections=connections,
                              connection_types=connection_types, recv_peers=children,
                              send_peers=children, data=data, scratch_size=scratch_size)
            node_types[tb_id] = 0
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]
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
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

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
            data_offsets[tb_id] = offset
            data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_offsets=data_offsets, data_sizes=data_sizes, scratch_size=scratch_size, 
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                node_types=node_types, nelem_per_send=nelem_per_send, 
                nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel


def allgather_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                     connections: dict, connection_types: dict, data: cp.ndarray,
                     allgather_length: int, nelem_per_send: int,
                     proxy_service: ProxyService = None):
    assert allgather_length % (k * group.nranks) == 0
    assert allgather_length == data.shape[0]
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
    nelem_per_chunk = allgather_length // total_chunks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
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
        (recv_sm_channels[tb_id], send_sm_channels[tb_id],
         recv_proxy_channels[tb_id], send_proxy_channels[tb_id]) = \
         make_channels(group=group, proxy_service=proxy_service, connections=connections,
                       connection_types=connection_types,
                       recv_peers=list(test_G.predecessors(group.my_rank)),
                       send_peers=children, data=data, scratch_size=None)
        assert len(recv_sm_channels[tb_id]) + len(recv_proxy_channels[tb_id]) <= 1
        node_types[tb_id] = 1
        data_offsets[tb_id] = offset
        data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_offsets=data_offsets, data_sizes=data_sizes, scratch_size=0, 
                recv_sm_scratches={}, recv_proxy_scratches={},
                node_types=node_types, nelem_per_send=nelem_per_send, 
                nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel


def reduce_scatter_kernel(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                          connections: dict, connection_types: dict, data: cp.ndarray,
                          reduce_scatter_length: int, nelem_per_send: int, scratch_size: int,
                          proxy_service: ProxyService = None):
    assert reduce_scatter_length % (k * group.nranks) == 0
    assert scratch_size >= nelem_per_send
    assert reduce_scatter_length == data.shape[0]
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
    nchunks_per_shard = k
    total_chunks = nchunks_per_shard * group.nranks
    nelem_per_chunk = reduce_scatter_length // total_chunks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    recv_sm_scratches, recv_proxy_scratches = {}, {}
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
        data_offsets[tb_id] = offset
        data_sizes[tb_id] = nelem_per_chunk * Cs[u, i]

    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_offsets=data_offsets, data_sizes=data_sizes, scratch_size=scratch_size, 
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                node_types=node_types, nelem_per_send=nelem_per_send, 
                nblocks=nblocks)
    kernel = PipelineKernel(**args)

    return kernel
