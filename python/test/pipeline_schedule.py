import struct
import cupy as cp
import networkx as nx
import os
import uuid
import math

from mscclpp import (
    ProxyService,
    Transport,
)
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder


# KERNEL_FILE = "pipeline_kernel.cu"
KERNEL_FILE = "pipeline_kernel_read.cu"
# KERNEL_FILE = "pipeline_kernel_no_divergence.cu"
# KERNEL_FILE = "pipeline_kernel_simplified_read.cu"

REDUCE_SCATTER_KERNEL_FILE = "pipeline_reduceScatter_kernel.cu"
# REDUCE_SCATTER_KERNEL_FILE = "pipeline_reduceScatter_kernel_sm_opt.cu"
# REDUCE_SCATTER_KERNEL_FILE = "pipeline_reduceScatter_kernel_coll_send.cu"

ALLGATHER_PARALLEL_SM_KERNEL_FILE = "pipeline_allgather_kernel_parallel_sm.cu"

REDUCE_SCATTER_PARALLEL_SM_KERNEL_FILE = "pipeline_reduceScatter_kernel_parallel_sm.cu"

# REDUCE_SCATTER_PARALLEL_SM_KERNEL_COLL_RE_FILE = "pipeline_reduceScatter_kernel_parallel_sm_coll_re.cu"
REDUCE_SCATTER_PARALLEL_SM_KERNEL_COLL_RE_FILE = "pipeline_reduceScatter_kernel_parallel_sm_coll_re_synced.cu"

REDUCE_SCATTER_PARALLEL_SM_KERNEL_HACK_FILE = "pipeline_reduceScatter_kernel_parallel_sm_hack.cu"

REDUCE_SCATTER_PARALLEL_SM_KERNEL_SENDTB_FILE = "pipeline_reduceScatter_kernel_parallel_sm_sendtb.cu"

MAX_NLOOPS = 1048576  # also defined in pipeline_reduceScatter_kernel.cu
MAX_NBLOCKS = 109


# Exception may not be triggered at all ranks.
# Different ranks may requre different num of threadblocks depending on parameters.
class ThreadBlockLimitException(Exception):
    def __init__(self, message, nblocks):
        super().__init__(message)
        self.nblocks = nblocks


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
        if nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={nblocks} > MAX_NBLOCKS", nblocks)
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
        self.kernel_file = KERNEL_FILE
        self.kernel_name = "pipeline_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
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
                    assert len(recv_sm_scratches[bid]) == len(recv_sm_channels[bid])
                if bid in recv_proxy_channels:
                    assert bid in recv_proxy_scratches
                    assert len(recv_proxy_scratches[bid]) == len(recv_proxy_channels[bid])

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


class AllgatherParallelSMPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peers of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peers of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        node_types: dict,     # node_types[tree]: <0: reduce node; =0: root node; >0: broadcast node.
        ntrees,
        n_parallel_sm_blocks: int = 1,
        nthreads=1024,
    ):
        n_peers = max([len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) for tree in range(ntrees)] +
                      [len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) for tree in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = ALLGATHER_PARALLEL_SM_KERNEL_FILE
        assert "parallel_sm" in self.kernel_file
        self.kernel_name = "pipeline_allgather_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nblocks = 0
        self.nthreads = nthreads
        self.data = data

        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        block_recv_sm_ch_starts = []
        block_send_sm_ch_starts = []
        block_recv_proxy_ch_starts = []
        block_send_proxy_ch_starts = []
        nrecvs_sm = []
        nsends_sm = []
        nrecvs_proxy = []
        nsends_proxy = []
        node_types_arr = []
        sm_block_idx_arr = []
        sm_block_cnt_arr = []
        sm_syncer_offset = 0
        sm_syncer_indics = []
        for tree in range(ntrees):
            assert (tree in recv_sm_channels or tree in send_sm_channels or 
                    tree in recv_proxy_channels or tree in send_proxy_channels)
            assert tree in data_chunk_offsets
            assert tree in data_chunk_sizes
            assert tree in node_types
            assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

            assert node_types[tree] > 0
            assert len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) <= 1

            nrecv_sm = len(recv_sm_channels.get(tree, []))
            nrecv_proxy = len(recv_proxy_channels.get(tree, []))
            assert nrecv_sm <= 1
            local_nblocks = max(nrecv_sm * n_parallel_sm_blocks, 1)
            self.nblocks += local_nblocks

            block_recv_sm_ch_starts += [len(recv_sm_handles_arr)] * local_nblocks
            block_recv_proxy_ch_starts += [len(recv_proxy_handles_arr)] * local_nblocks
            recv_sm_handles = [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
            recv_proxy_handles = [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
            assert len(recv_sm_handles) + len(recv_proxy_handles) <= 1
            recv_sm_handles_arr += recv_sm_handles
            recv_proxy_handles_arr += recv_proxy_handles
            nrecvs_sm += [nrecv_sm] * local_nblocks
            nrecvs_proxy += [nrecv_proxy] * local_nblocks

            block_send_sm_ch_starts += [len(send_sm_handles_arr)] * local_nblocks
            block_send_proxy_ch_starts += [len(send_proxy_handles_arr)] * local_nblocks
            send_sm_handles = [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
            send_proxy_handles = [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]
            assert len(send_sm_handles) + len(send_proxy_handles) <= n_peers
            send_sm_handles_arr += send_sm_handles
            send_proxy_handles_arr += send_proxy_handles
            nsends_sm += [len(send_sm_handles)] * local_nblocks
            nsends_proxy += [len(send_proxy_handles)] * local_nblocks
            node_types_arr += [node_types[tree]] * local_nblocks

            if nrecv_sm > 0:
                assert nrecv_proxy == 0
                assert local_nblocks == n_parallel_sm_blocks
                sm_block_idx_arr += list(range(n_parallel_sm_blocks)) * nrecv_sm
                sm_block_cnt_arr += [n_parallel_sm_blocks] * nrecv_sm * n_parallel_sm_blocks
                for i in range(nrecv_sm):
                    sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                    sm_syncer_offset += 1
            else:
                sm_block_idx_arr += [0]
                sm_block_cnt_arr += [0]
                sm_syncer_indics += [None]                

            self.data_chunk_offsets += [data_chunk_offsets[tree]] * local_nblocks
            self.data_chunk_sizes += [data_chunk_sizes[tree]] * local_nblocks
        
        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

        recv_sm_handles_mem = cp.asarray(memoryview(b"".join(recv_sm_handles_arr)), dtype=cp.uint8)
        send_sm_handles_mem = cp.asarray(memoryview(b"".join(send_sm_handles_arr)), dtype=cp.uint8)
        recv_proxy_handles_mem = cp.asarray(memoryview(b"".join(recv_proxy_handles_arr)), dtype=cp.uint8)
        send_proxy_handles_mem = cp.asarray(memoryview(b"".join(send_proxy_handles_arr)), dtype=cp.uint8)
        assert len(recv_sm_handles_arr) > 0 or recv_sm_handles_mem.data.ptr == 0
        assert len(send_sm_handles_arr) > 0 or send_sm_handles_mem.data.ptr == 0
        assert len(recv_proxy_handles_arr) > 0 or recv_proxy_handles_mem.data.ptr == 0
        assert len(send_proxy_handles_arr) > 0 or send_proxy_handles_mem.data.ptr == 0
        block_recv_sm_ch_starts = cp.array(block_recv_sm_ch_starts, dtype=cp.int32)
        block_send_sm_ch_starts = cp.array(block_send_sm_ch_starts, dtype=cp.int32)
        block_recv_proxy_ch_starts = cp.array(block_recv_proxy_ch_starts, dtype=cp.int32)
        block_send_proxy_ch_starts = cp.array(block_send_proxy_ch_starts, dtype=cp.int32)
        nrecvs_sm = cp.array(nrecvs_sm, dtype=cp.int32)
        nsends_sm = cp.array(nsends_sm, dtype=cp.int32)
        nrecvs_proxy = cp.array(nrecvs_proxy, dtype=cp.int32)
        nsends_proxy = cp.array(nsends_proxy, dtype=cp.int32)
        node_types_arr = cp.array(node_types_arr, dtype=cp.byte)
        sm_block_idx_arr = cp.array(sm_block_idx_arr, dtype=cp.int32)
        sm_block_cnt_arr = cp.array(sm_block_cnt_arr, dtype=cp.int32)
        sm_syncer_num = sm_syncer_offset
        sm_syncer_arr = cp.empty(sm_syncer_num * 12, dtype=cp.bool_)
        sm_syncer_ptr_arr = [struct.pack("P", sm_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in sm_syncer_indics]
        sm_syncer_arr_mem = cp.asarray(memoryview(b"".join(sm_syncer_ptr_arr)), dtype=cp.uint8)

        assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert block_recv_sm_ch_starts.shape[0] == self.nblocks and block_send_sm_ch_starts.shape[0] == self.nblocks
        assert block_recv_proxy_ch_starts.shape[0] == self.nblocks and block_send_proxy_ch_starts.shape[0] == self.nblocks
        assert nrecvs_sm.shape[0] == self.nblocks and nsends_sm.shape[0] == self.nblocks
        assert nrecvs_proxy.shape[0] == self.nblocks and nsends_proxy.shape[0] == self.nblocks
        assert node_types_arr.shape[0] == self.nblocks
        assert sm_syncer_num == n_recv_sm_channels
        assert sm_block_idx_arr.shape[0] == self.nblocks
        assert sm_block_cnt_arr.shape[0] == self.nblocks
        assert len(sm_syncer_ptr_arr) == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", 0)
        self.params += struct.pack("P", block_recv_sm_ch_starts.data.ptr) + struct.pack("P", block_send_sm_ch_starts.data.ptr)
        self.params += struct.pack("P", block_recv_proxy_ch_starts.data.ptr) + struct.pack("P", block_send_proxy_ch_starts.data.ptr)
        self.params += struct.pack("P", 0)
        self.params += struct.pack("P", nrecvs_sm.data.ptr) + struct.pack("P", nsends_sm.data.ptr)
        self.params += struct.pack("P", nrecvs_proxy.data.ptr) + struct.pack("P", nsends_proxy.data.ptr)
        self.params += struct.pack("P", node_types_arr.data.ptr) + struct.pack("Q", 0) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", sm_block_idx_arr.data.ptr) + struct.pack("P", sm_block_cnt_arr.data.ptr)
        self.params += struct.pack("P", sm_syncer_arr_mem.data.ptr)

        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, block_recv_sm_ch_starts, block_send_sm_ch_starts,
                      block_recv_proxy_ch_starts, block_send_proxy_ch_starts,
                      nrecvs_sm, nsends_sm, nrecvs_proxy, nsends_proxy,
                      node_types_arr, sm_block_idx_arr, sm_block_cnt_arr,
                      sm_syncer_arr, sm_syncer_ptr_arr, sm_syncer_arr_mem]
        self._data_starts_nelem_totals = {}
        self._params = {}
    

    def prepare_params(self, nelem_total, nelem_per_send):
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


class ReduceScatterPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peer of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peer of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        ntrees: int,
        nthreads=1024,
    ):
        n_peers = max([len(recv_sm_channels.get(t, [])) + len(recv_proxy_channels.get(t, [])) for t in range(ntrees)] +
                      [len(send_sm_channels.get(t, [])) + len(send_proxy_channels.get(t, [])) for t in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = REDUCE_SCATTER_KERNEL_FILE
        self.kernel_name = "pipeline_reduceScatter_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nthreads = nthreads
        self.data = data
    
        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        recv_sm_channel_indics = []
        send_sm_channel_indics = []
        recv_proxy_channel_indics = []
        send_proxy_channel_indics = []
        recv_scratches_arr = []
        reduce_counts_arr = []
        nrecv_peers_arr = []
        send_status_offset = 0
        send_status_indics = []
        first_block_arr = []
        self.nblocks = 0
        null_buf = cp.empty(0, dtype=cp.int32)
        assert null_buf.data.ptr == 0
        for tree in range(ntrees):
            assert (tree in recv_sm_channels or tree in send_sm_channels or 
                    tree in recv_proxy_channels or tree in send_proxy_channels)
            assert tree in data_chunk_offsets
            assert tree in data_chunk_sizes
            assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

            if tree in recv_sm_channels:
                assert tree in recv_sm_scratches
                assert len(recv_sm_scratches[tree]) == len(recv_sm_channels[tree])
            if tree in recv_proxy_channels:
                assert tree in recv_proxy_scratches
                assert len(recv_proxy_scratches[tree]) == len(recv_proxy_channels[tree])

            nrecv_sm = len(recv_sm_channels.get(tree, []))
            nrecv_proxy = len(recv_proxy_channels.get(tree, []))
            nrecv_peers = nrecv_sm + nrecv_proxy
            assert nrecv_peers <= n_peers
            assert len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) <= 1
            send_sm = len(send_sm_channels.get(tree, [])) > 0
            send_proxy = len(send_proxy_channels.get(tree, [])) > 0
            assert send_sm or send_proxy or nrecv_peers > 0

            if nrecv_peers > 0:
                self.nblocks += nrecv_peers
                recv_sm_channel_indics += [len(recv_sm_handles_arr) + i for i in range(nrecv_sm)] + [-1] * nrecv_proxy
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1] * nrecv_peers
                recv_proxy_channel_indics += [-1] * nrecv_sm + [len(recv_proxy_handles_arr) + i for i in range(nrecv_proxy)]
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1] * nrecv_peers

                recv_sm_handles_arr += [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
                send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
                recv_proxy_handles_arr += [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
                send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

                assert len(recv_sm_scratches.get(tree, [])) == nrecv_sm
                assert len(recv_proxy_scratches.get(tree, [])) == nrecv_proxy
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_sm_scratches.get(tree, [])] + \
                                      [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_proxy_scratches.get(tree, [])]
                
                reduce_counts = cp.empty(MAX_NLOOPS, dtype=cp.int32)
                reduce_counts_arr += [reduce_counts] * nrecv_peers
                nrecv_peers_arr += [nrecv_peers] * nrecv_peers

                if send_proxy:
                    send_status_indics += [send_status_offset] * nrecv_peers
                    send_status_offset += 1
                else:
                    send_status_indics += [None] * nrecv_peers

                first_block_arr += [True] + [False] * (nrecv_peers - 1)

                self.data_chunk_offsets += [data_chunk_offsets[tree]] * nrecv_peers
                self.data_chunk_sizes += [data_chunk_sizes[tree]] * nrecv_peers
            else:
                self.nblocks += 1
                recv_sm_channel_indics += [-1]
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1]
                recv_proxy_channel_indics += [-1]
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1]

                send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
                send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

                assert tree not in recv_sm_scratches
                assert tree not in recv_proxy_scratches
                recv_scratches_arr += [struct.pack("P", 0)]

                reduce_counts_arr += [null_buf]
                nrecv_peers_arr += [0]

                if send_proxy:
                    send_status_indics += [send_status_offset]
                    send_status_offset += 1
                else:
                    send_status_indics += [None]

                first_block_arr += [True]

                self.data_chunk_offsets += [data_chunk_offsets[tree]]
                self.data_chunk_sizes += [data_chunk_sizes[tree]]

        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

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
        recv_sm_channel_indics = cp.array(recv_sm_channel_indics, dtype=cp.int32)
        send_sm_channel_indics = cp.array(send_sm_channel_indics, dtype=cp.int32)
        recv_proxy_channel_indics = cp.array(recv_proxy_channel_indics, dtype=cp.int32)
        send_proxy_channel_indics = cp.array(send_proxy_channel_indics, dtype=cp.int32)
        reduce_counts_ptr_arr = [struct.pack("P", arr.data.ptr) for arr in reduce_counts_arr]
        reduce_counts_arr_mem = cp.asarray(memoryview(b"".join(reduce_counts_ptr_arr)), dtype=cp.uint8)
        nrecv_peers_arr = cp.array(nrecv_peers_arr, dtype=cp.int32)

        if "coll_send" in self.kernel_file:
            send_status_mem_size = send_status_offset
            send_status_arr = cp.empty(send_status_mem_size, dtype=cp.uint64)
            send_status_ptr_arr = [struct.pack("P", send_status_arr.data.ptr + i * 8) if i is not None else struct.pack("P", 0) for i in send_status_indics]
            send_status_arr_mem = cp.asarray(memoryview(b"".join(send_status_ptr_arr)), dtype=cp.uint8)
        else:
            send_status_arr, send_status_ptr_arr, send_status_arr_mem = None, None, None

        first_block_arr = cp.array(first_block_arr, dtype=cp.bool_)

        assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(recv_scratches_arr) == self.nblocks
        assert recv_sm_channel_indics.shape[0] == self.nblocks
        assert send_sm_channel_indics.shape[0] == self.nblocks
        assert recv_proxy_channel_indics.shape[0] == self.nblocks
        assert send_proxy_channel_indics.shape[0] == self.nblocks
        assert len(reduce_counts_arr) == self.nblocks
        assert nrecv_peers_arr.shape[0] == self.nblocks
        assert send_status_ptr_arr is None or len(send_status_ptr_arr) == self.nblocks
        assert first_block_arr.shape[0] == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_sm_channel_indics.data.ptr) + struct.pack("P", send_sm_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_proxy_channel_indics.data.ptr) + struct.pack("P", send_proxy_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_scratches_mem.data.ptr) + struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", reduce_counts_arr_mem.data.ptr) + struct.pack("P", nrecv_peers_arr.data.ptr)
        if "coll_send" in self.kernel_file:
            self.params += struct.pack("P", send_status_arr_mem.data.ptr)
        self.params += struct.pack("P", first_block_arr.data.ptr)
        
        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches, recv_scratches_mem, recv_scratches_arr,
                      recv_sm_channel_indics, send_sm_channel_indics,
                      recv_proxy_channel_indics, send_proxy_channel_indics,
                      reduce_counts_arr, nrecv_peers_arr, send_status_arr,
                      reduce_counts_ptr_arr, send_status_ptr_arr,
                      reduce_counts_arr_mem, send_status_arr_mem,
                      first_block_arr]
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
            assert all(math.ceil(self.data_chunk_sizes[bid] * nelem_per_chunk / nelem_per_send) <= MAX_NLOOPS for bid in range(self.nblocks))
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None, debug_flag=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        params += struct.pack("Q", debug_flag) if debug_flag is not None else struct.pack("Q", 0)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None, debug_flag=None):
        return self.get_func(nelem_total, nelem_per_send, debug_flag)(stream_ptr)


class ReduceScatterParallelSMPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peer of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peer of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        ntrees: int,
        n_parallel_sm_blocks: int = 1,
        leaf_nodes: dict = None,
        skip_leaf_tb: bool = False,
        nthreads=1024,
    ):
        assert (leaf_nodes is not None) == skip_leaf_tb
        n_peers = max([len(recv_sm_channels.get(t, [])) + len(recv_proxy_channels.get(t, [])) for t in range(ntrees)] +
                      [len(send_sm_channels.get(t, [])) + len(send_proxy_channels.get(t, [])) for t in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = REDUCE_SCATTER_PARALLEL_SM_KERNEL_FILE
        assert "parallel_sm" in self.kernel_file
        self.kernel_name = "pipeline_reduceScatter_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nthreads = nthreads
        self.data = data
    
        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        recv_sm_channel_indics = []
        send_sm_channel_indics = []
        recv_proxy_channel_indics = []
        send_proxy_channel_indics = []
        recv_scratches_arr = []
        reduce_counts_arr = []
        nrecv_peers_arr = []
        send_status_offset = 0
        send_status_indics = []
        first_block_arr = []
        sm_block_idx_arr = []
        sm_block_cnt_arr = []
        sm_syncer_offset = 0
        sm_syncer_indics = []
        skip_signal_arr = []
        self.nblocks = 0
        null_buf = cp.empty(0, dtype=cp.int32)
        assert null_buf.data.ptr == 0
        for tree in range(ntrees):
            assert (tree in recv_sm_channels or tree in send_sm_channels or 
                    tree in recv_proxy_channels or tree in send_proxy_channels)
            assert tree in data_chunk_offsets
            assert tree in data_chunk_sizes
            assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

            if tree in recv_sm_channels:
                assert tree in recv_sm_scratches
                assert len(recv_sm_scratches[tree]) == len(recv_sm_channels[tree])
            if tree in recv_proxy_channels:
                assert tree in recv_proxy_scratches
                assert len(recv_proxy_scratches[tree]) == len(recv_proxy_channels[tree])

            nrecv_sm = len(recv_sm_channels.get(tree, []))
            nrecv_proxy = len(recv_proxy_channels.get(tree, []))
            nrecv_peers = nrecv_sm + nrecv_proxy
            assert nrecv_peers <= n_peers
            local_nblocks = nrecv_sm * n_parallel_sm_blocks + nrecv_proxy
            assert len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) <= 1
            send_sm = len(send_sm_channels.get(tree, [])) > 0
            send_proxy = len(send_proxy_channels.get(tree, [])) > 0
            assert send_sm or send_proxy or nrecv_peers > 0

            if nrecv_peers > 0:
                self.nblocks += local_nblocks
                for i in range(nrecv_sm):
                    recv_sm_channel_indics += [len(recv_sm_handles_arr) + i] * n_parallel_sm_blocks
                    if leaf_nodes is not None:
                        skip_signal_arr += [leaf_nodes[tree][i]]  * n_parallel_sm_blocks
                    else:
                        skip_signal_arr += [False] * n_parallel_sm_blocks
                recv_sm_channel_indics += [-1] * nrecv_proxy
                skip_signal_arr += [False] * nrecv_proxy
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1] * local_nblocks
                recv_proxy_channel_indics += [-1] * nrecv_sm * n_parallel_sm_blocks + [len(recv_proxy_handles_arr) + i for i in range(nrecv_proxy)]
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1] * local_nblocks

                recv_sm_handles_arr += [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
                send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
                recv_proxy_handles_arr += [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
                send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

                assert len(recv_sm_scratches.get(tree, [])) == nrecv_sm
                assert len(recv_proxy_scratches.get(tree, [])) == nrecv_proxy
                for scratch_buff in recv_sm_scratches.get(tree, []):
                    recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr)] * n_parallel_sm_blocks
                recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_proxy_scratches.get(tree, [])]
                
                reduce_counts = cp.empty(MAX_NLOOPS, dtype=cp.int32)
                reduce_counts_arr += [reduce_counts] * local_nblocks
                nrecv_peers_arr += [nrecv_peers] * local_nblocks

                if send_proxy:
                    send_status_indics += [send_status_offset] * local_nblocks
                    send_status_offset += 1
                else:
                    send_status_indics += [None] * local_nblocks

                first_block_arr += [True] + [False] * (local_nblocks - 1)

                sm_block_idx_arr += list(range(n_parallel_sm_blocks)) * nrecv_sm + [0] * nrecv_proxy
                sm_block_cnt_arr += [n_parallel_sm_blocks] * nrecv_sm * n_parallel_sm_blocks + [0] * nrecv_proxy
                for i in range(nrecv_sm):
                    sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                    sm_syncer_offset += 1
                sm_syncer_indics += [None] * nrecv_proxy

                self.data_chunk_offsets += [data_chunk_offsets[tree]] * local_nblocks
                self.data_chunk_sizes += [data_chunk_sizes[tree]] * local_nblocks
            elif leaf_nodes is None or send_proxy:
                self.nblocks += 1
                recv_sm_channel_indics += [-1]
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1]
                recv_proxy_channel_indics += [-1]
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1]

                send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
                send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

                assert tree not in recv_sm_scratches
                assert tree not in recv_proxy_scratches
                recv_scratches_arr += [struct.pack("P", 0)]

                reduce_counts_arr += [null_buf]
                nrecv_peers_arr += [0]

                if send_proxy:
                    send_status_indics += [send_status_offset]
                    send_status_offset += 1
                else:
                    send_status_indics += [None]

                first_block_arr += [True]

                sm_block_idx_arr += [0]
                sm_block_cnt_arr += [0]
                sm_syncer_indics += [None]

                assert leaf_nodes is None or send_proxy
                skip_signal_arr += [False]

                self.data_chunk_offsets += [data_chunk_offsets[tree]]
                self.data_chunk_sizes += [data_chunk_sizes[tree]]

        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

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
        recv_sm_channel_indics = cp.array(recv_sm_channel_indics, dtype=cp.int32)
        send_sm_channel_indics = cp.array(send_sm_channel_indics, dtype=cp.int32)
        recv_proxy_channel_indics = cp.array(recv_proxy_channel_indics, dtype=cp.int32)
        send_proxy_channel_indics = cp.array(send_proxy_channel_indics, dtype=cp.int32)
        reduce_counts_ptr_arr = [struct.pack("P", arr.data.ptr) for arr in reduce_counts_arr]
        reduce_counts_arr_mem = cp.asarray(memoryview(b"".join(reduce_counts_ptr_arr)), dtype=cp.uint8)
        nrecv_peers_arr = cp.array(nrecv_peers_arr, dtype=cp.int32)

        assert  "coll_send" not in self.kernel_file
        if "coll_send" in self.kernel_file:
            send_status_mem_size = send_status_offset
            send_status_arr = cp.empty(send_status_mem_size, dtype=cp.uint64)
            send_status_ptr_arr = [struct.pack("P", send_status_arr.data.ptr + i * 8) if i is not None else struct.pack("P", 0) for i in send_status_indics]
            send_status_arr_mem = cp.asarray(memoryview(b"".join(send_status_ptr_arr)), dtype=cp.uint8)
        else:
            send_status_arr, send_status_ptr_arr, send_status_arr_mem = None, None, None

        first_block_arr = cp.array(first_block_arr, dtype=cp.bool_)
        
        sm_block_idx_arr = cp.array(sm_block_idx_arr, dtype=cp.int32)
        sm_block_cnt_arr = cp.array(sm_block_cnt_arr, dtype=cp.int32)

        sm_syncer_num = sm_syncer_offset
        sm_syncer_arr = cp.empty(sm_syncer_num * 12, dtype=cp.bool_)
        sm_syncer_ptr_arr = [struct.pack("P", sm_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in sm_syncer_indics]
        sm_syncer_arr_mem = cp.asarray(memoryview(b"".join(sm_syncer_ptr_arr)), dtype=cp.uint8)
        skip_signal_arr = cp.array(skip_signal_arr, dtype=cp.bool_)

        if leaf_nodes is None:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        else:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels
            assert len(send_sm_handles_arr) == sum(0 if len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) == 0
                                                   else len(send_sm_channels.get(tree, [])) for tree in range(ntrees))
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(recv_scratches_arr) == self.nblocks
        assert recv_sm_channel_indics.shape[0] == self.nblocks
        assert send_sm_channel_indics.shape[0] == self.nblocks
        assert recv_proxy_channel_indics.shape[0] == self.nblocks
        assert send_proxy_channel_indics.shape[0] == self.nblocks
        assert len(reduce_counts_arr) == self.nblocks
        assert nrecv_peers_arr.shape[0] == self.nblocks
        assert send_status_ptr_arr is None or len(send_status_ptr_arr) == self.nblocks
        assert first_block_arr.shape[0] == self.nblocks
        assert sm_block_idx_arr.shape[0] == self.nblocks
        assert sm_block_cnt_arr.shape[0] == self.nblocks
        assert len(sm_syncer_ptr_arr) == self.nblocks
        assert skip_signal_arr.shape[0] == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_sm_channel_indics.data.ptr) + struct.pack("P", send_sm_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_proxy_channel_indics.data.ptr) + struct.pack("P", send_proxy_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_scratches_mem.data.ptr) + struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", reduce_counts_arr_mem.data.ptr) + struct.pack("P", nrecv_peers_arr.data.ptr)
        if "coll_send" in self.kernel_file:
            self.params += struct.pack("P", send_status_arr_mem.data.ptr)
        self.params += struct.pack("P", first_block_arr.data.ptr)
        self.params += struct.pack("P", sm_block_idx_arr.data.ptr) + struct.pack("P", sm_block_cnt_arr.data.ptr)
        self.params += struct.pack("P", sm_syncer_arr_mem.data.ptr) + struct.pack("P", skip_signal_arr.data.ptr)

        
        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches, recv_scratches_mem, recv_scratches_arr,
                      recv_sm_channel_indics, send_sm_channel_indics,
                      recv_proxy_channel_indics, send_proxy_channel_indics,
                      reduce_counts_arr, nrecv_peers_arr, send_status_arr,
                      reduce_counts_ptr_arr, send_status_ptr_arr,
                      reduce_counts_arr_mem, send_status_arr_mem,
                      first_block_arr, skip_signal_arr,
                      sm_block_idx_arr, sm_block_cnt_arr,
                      sm_syncer_arr, sm_syncer_ptr_arr, sm_syncer_arr_mem]
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
            assert all(math.ceil(self.data_chunk_sizes[bid] * nelem_per_chunk / nelem_per_send) <= MAX_NLOOPS for bid in range(self.nblocks))
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None, debug_flag=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        params += struct.pack("Q", debug_flag) if debug_flag is not None else struct.pack("Q", 0)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None, debug_flag=None):
        return self.get_func(nelem_total, nelem_per_send, debug_flag)(stream_ptr)


class ReduceScatterParallelSMCollREPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peer of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peer of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        ntrees: int,
        n_parallel_sm_blocks: int = 1,
        n_parallel_reduce_blocks: int = None,
        leaf_nodes: dict = None,
        skip_leaf_tb: bool = False,
        pipeline_groups: list = None,  # list of list
        nthreads=1024,
    ):
        assert (leaf_nodes is not None) == skip_leaf_tb
        n_peers = max([len(recv_sm_channels.get(t, [])) + len(recv_proxy_channels.get(t, [])) for t in range(ntrees)] +
                      [len(send_sm_channels.get(t, [])) + len(send_proxy_channels.get(t, [])) for t in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = REDUCE_SCATTER_PARALLEL_SM_KERNEL_COLL_RE_FILE
        assert "parallel_sm" in self.kernel_file
        if n_parallel_reduce_blocks is not None and n_parallel_reduce_blocks > 0:
            assert "synced" in self.kernel_file
        if pipeline_groups is not None:
            assert "synced" in self.kernel_file
            assert ntrees % len(pipeline_groups) == 0
            assert all(len(g) == ntrees // len(pipeline_groups) for g in pipeline_groups)
        else:
            pipeline_groups = [[i] for i in range(ntrees)]
        self.kernel_name = "pipeline_reduceScatter_coll_re_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nthreads = nthreads
        self.data = data
    
        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_scratch_arr_save = []
        received_arr_save = []
        reduce_arr_save = []
        reduce_or_get_save = []
        start_signal_save = []

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        recv_sm_channel_indics = []
        send_sm_channel_indics = []
        recv_proxy_channel_indics = []
        send_proxy_channel_indics = []
        recv_scratch_arr_block = []
        nrecv_sm_block = []
        nrecv_proxy_block = []
        nrecv_peers_block = []
        reduce_block_idx_block = []
        reduce_block_cnt_block = []
        reduce_syncer_offset = 0
        reduce_syncer_indics = []
        received_arr_block = []
        reduce_arr_block = []
        sm_block_idx_block = []
        sm_block_cnt_block = []
        sm_syncer_offset = 0
        sm_syncer_indics = []
        start_signal_block = []
        next_start_signal_block = []
        reduce_or_get_block = []
        skip_signal_block = []
        self.nblocks = 0
        null_buf = cp.empty(0, dtype=cp.int32)
        assert null_buf.data.ptr == 0

        for pp_group in pipeline_groups:
            next_start_signal_ptr = struct.pack("P", 0)
            for idx, tree in enumerate(pp_group):
                start_signal_ptr = next_start_signal_ptr
                next_start_signal = cp.zeros(1, dtype=cp.int32) if idx < len(pp_group) - 1 else null_buf
                start_signal_save.append(next_start_signal)
                next_start_signal_ptr = struct.pack("P", next_start_signal.data.ptr)

                assert (tree in recv_sm_channels or tree in send_sm_channels or 
                        tree in recv_proxy_channels or tree in send_proxy_channels)
                assert tree in data_chunk_offsets
                assert tree in data_chunk_sizes
                assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

                if tree in recv_sm_channels:
                    assert tree in recv_sm_scratches
                    assert len(recv_sm_scratches[tree]) == len(recv_sm_channels[tree])
                if tree in recv_proxy_channels:
                    assert tree in recv_proxy_scratches
                    assert len(recv_proxy_scratches[tree]) == len(recv_proxy_channels[tree])

                nrecv_sm = len(recv_sm_channels.get(tree, []))
                nrecv_proxy = len(recv_proxy_channels.get(tree, []))
                nrecv_peers = nrecv_sm + nrecv_proxy
                assert nrecv_peers <= n_peers
                assert len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) <= 1
                send_sm = len(send_sm_channels.get(tree, [])) > 0
                send_proxy = len(send_proxy_channels.get(tree, [])) > 0
                assert nrecv_peers > 0 or send_sm or send_proxy

                if nrecv_peers == 0 and not send_proxy and leaf_nodes is not None:
                    continue

                n_parallel_blocks = n_parallel_sm_blocks if nrecv_peers > 0 else 1
                local_nblocks = max(1, nrecv_sm) * n_parallel_blocks
                n_reduce_blocks = 0 if n_parallel_reduce_blocks is None or nrecv_peers == 0 or (nrecv_peers == nrecv_sm == 1) else n_parallel_reduce_blocks
                local_nblocks += n_reduce_blocks
                self.nblocks += local_nblocks
                if nrecv_sm > 0:
                    for i in range(nrecv_sm):
                        recv_sm_channel_indics += [len(recv_sm_handles_arr) + i] * n_parallel_sm_blocks
                        if leaf_nodes is not None:
                            skip_signal_block += [leaf_nodes[tree][i]] * n_parallel_sm_blocks
                        else:
                            skip_signal_block += [False] * n_parallel_sm_blocks
                    recv_sm_channel_indics += [-1] * n_reduce_blocks
                    skip_signal_block += [False] * n_reduce_blocks
                else:
                    recv_sm_channel_indics += [-1] * local_nblocks
                    skip_signal_block += [False] * local_nblocks
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1] + [-1] * (local_nblocks - 1)
                recv_proxy_channel_indics += [len(recv_proxy_handles_arr) if nrecv_proxy > 0 else -1] + [-1] * (local_nblocks - 1)
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1] + [-1] * (local_nblocks - 1)

                recv_sm_handles_arr += [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
                send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
                recv_proxy_handles_arr += [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
                send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

                assert len(recv_sm_scratches.get(tree, [])) == nrecv_sm
                assert len(recv_proxy_scratches.get(tree, [])) == nrecv_proxy
                recv_scratch_arr = [struct.pack("P", buff.data.ptr) for buff in recv_sm_scratches.get(tree, [])] + \
                                [struct.pack("P", buff.data.ptr) for buff in recv_proxy_scratches.get(tree, [])]
                recv_scratch_arr_mem = cp.asarray(memoryview(b"".join(recv_scratch_arr)), dtype=cp.uint8)
                recv_scratch_arr_block += [struct.pack("P", recv_scratch_arr_mem.data.ptr)] * local_nblocks
                recv_scratch_arr_save.append((recv_scratch_arr, recv_scratch_arr_mem))

                nrecv_sm_block += [nrecv_sm] * local_nblocks
                nrecv_proxy_block += [nrecv_proxy] + [0] * (local_nblocks - 1)
                nrecv_peers_block += [nrecv_peers] * local_nblocks

                reduce_block_idx_block += list(range(local_nblocks))
                reduce_block_cnt_block += [local_nblocks] * local_nblocks
                reduce_syncer_indics += [reduce_syncer_offset] * local_nblocks
                reduce_syncer_offset += 1

                if nrecv_peers <= 1:
                    received_arr_block += [struct.pack("P", 0)] * local_nblocks
                else:
                    received_arr = cp.zeros(nrecv_peers, dtype=cp.int32)
                    received_arr_block += [struct.pack("P", received_arr.data.ptr)] * local_nblocks
                    received_arr_save.append((received_arr))

                if nrecv_sm <= 1:
                    reduce_arr_block += [struct.pack("P", 0)] * local_nblocks
                else:
                    reduce_arr = cp.zeros(nrecv_sm, dtype=cp.int32)
                    reduce_arr_block += [struct.pack("P", reduce_arr.data.ptr)] * local_nblocks
                    reduce_arr_save.append((reduce_arr))

                sm_block_idx_block += list(range(n_parallel_blocks)) * max(1, nrecv_sm) + [-1] * n_reduce_blocks
                sm_block_cnt_block += [n_parallel_sm_blocks if nrecv_peers > 0 else 1] * (local_nblocks - n_reduce_blocks) + [0] * n_reduce_blocks
                if nrecv_peers > 0:
                    for i in range(max(1, nrecv_sm)):
                        sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                        sm_syncer_offset += 1
                    sm_syncer_indics += [None] * n_reduce_blocks
                    reduce_or_get = cp.empty(max(1, nrecv_sm), dtype=cp.int32)
                    reduce_or_get_save.append(reduce_or_get)
                    for i in range(max(1, nrecv_sm)):
                        reduce_or_get_block += [struct.pack("P", reduce_or_get.data.ptr + i * 4)] * n_parallel_sm_blocks
                    reduce_or_get_block += [struct.pack("P", 0)] * n_reduce_blocks
                else:
                    assert n_parallel_blocks == 1
                    sm_syncer_indics += [None]
                    reduce_or_get_block += [struct.pack("P", 0)]
                
                start_signal_block += [start_signal_ptr] * local_nblocks
                next_start_signal_block += [next_start_signal_ptr] * local_nblocks

                self.data_chunk_offsets += [data_chunk_offsets[tree]] * local_nblocks
                self.data_chunk_sizes += [data_chunk_sizes[tree]] * local_nblocks

        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

        recv_sm_handles_mem = cp.asarray(memoryview(b"".join(recv_sm_handles_arr)), dtype=cp.uint8)
        send_sm_handles_mem = cp.asarray(memoryview(b"".join(send_sm_handles_arr)), dtype=cp.uint8)
        recv_proxy_handles_mem = cp.asarray(memoryview(b"".join(recv_proxy_handles_arr)), dtype=cp.uint8)
        send_proxy_handles_mem = cp.asarray(memoryview(b"".join(send_proxy_handles_arr)), dtype=cp.uint8)
        recv_scratch_arr_mem = cp.asarray(memoryview(b"".join(recv_scratch_arr_block)), dtype=cp.uint8)
        received_arr_mem = cp.asarray(memoryview(b"".join(received_arr_block)), dtype=cp.uint8)
        reduce_arr_mem = cp.asarray(memoryview(b"".join(reduce_arr_block)), dtype=cp.uint8)
        reduce_or_get_mem = cp.asarray(memoryview(b"".join(reduce_or_get_block)), dtype=cp.uint8)
        start_signal_mem = cp.asarray(memoryview(b"".join(start_signal_block)), dtype=cp.uint8)
        next_start_signal_mem = cp.asarray(memoryview(b"".join(next_start_signal_block)), dtype=cp.uint8)
        assert len(recv_sm_handles_arr) > 0 or recv_sm_handles_mem.data.ptr == 0
        assert len(send_sm_handles_arr) > 0 or send_sm_handles_mem.data.ptr == 0
        assert len(recv_proxy_handles_arr) > 0 or recv_proxy_handles_mem.data.ptr == 0
        assert len(send_proxy_handles_arr) > 0 or send_proxy_handles_mem.data.ptr == 0
        assert len(recv_scratch_arr_block) > 0 or recv_scratch_arr_mem.data.ptr == 0
        assert len(received_arr_block) > 0 or received_arr_mem.data.ptr == 0
        assert len(reduce_arr_block) > 0 or reduce_arr_mem.data.ptr == 0
        assert len(reduce_or_get_block) > 0 or reduce_or_get_mem.data.ptr == 0
        assert len(start_signal_block) > 0 or start_signal_mem.data.ptr == 0
        assert len(next_start_signal_block) > 0 or next_start_signal_mem.data.ptr == 0
        recv_sm_channel_indics = cp.array(recv_sm_channel_indics, dtype=cp.int32)
        send_sm_channel_indics = cp.array(send_sm_channel_indics, dtype=cp.int32)
        recv_proxy_channel_indics = cp.array(recv_proxy_channel_indics, dtype=cp.int32)
        send_proxy_channel_indics = cp.array(send_proxy_channel_indics, dtype=cp.int32)
        nrecv_sm_block = cp.array(nrecv_sm_block, dtype=cp.int32)
        nrecv_proxy_block = cp.array(nrecv_proxy_block, dtype=cp.int32)
        nrecv_peers_block = cp.array(nrecv_peers_block, dtype=cp.int32)
        
        reduce_block_idx_block = cp.array(reduce_block_idx_block, dtype=cp.int32)
        reduce_block_cnt_block = cp.array(reduce_block_cnt_block, dtype=cp.int32)

        reduce_syncer_num = reduce_syncer_offset
        reduce_syncer_arr = cp.zeros(reduce_syncer_num * 12, dtype=cp.bool_)
        reduce_syncer_ptr_arr = [struct.pack("P", reduce_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in reduce_syncer_indics]
        reduce_syncer_arr_mem = cp.asarray(memoryview(b"".join(reduce_syncer_ptr_arr)), dtype=cp.uint8)
        
        sm_block_idx_block = cp.array(sm_block_idx_block, dtype=cp.int32)
        sm_block_cnt_block = cp.array(sm_block_cnt_block, dtype=cp.int32)

        sm_syncer_num = sm_syncer_offset
        sm_syncer_arr = cp.zeros(sm_syncer_num * 12, dtype=cp.bool_)
        sm_syncer_ptr_arr = [struct.pack("P", sm_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in sm_syncer_indics]
        sm_syncer_arr_mem = cp.asarray(memoryview(b"".join(sm_syncer_ptr_arr)), dtype=cp.uint8)
        skip_signal_block = cp.array(skip_signal_block, dtype=cp.bool_)

        if leaf_nodes is None:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        else:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels
            assert len(send_sm_handles_arr) == sum(0 if len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) == 0
                                                   else len(send_sm_channels.get(tree, [])) for tree in range(ntrees))
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert recv_sm_channel_indics.shape[0] == self.nblocks
        assert send_sm_channel_indics.shape[0] == self.nblocks
        assert recv_proxy_channel_indics.shape[0] == self.nblocks
        assert send_proxy_channel_indics.shape[0] == self.nblocks
        assert len(recv_scratch_arr_block) == self.nblocks
        assert nrecv_sm_block.shape[0] == self.nblocks
        assert nrecv_proxy_block.shape[0] == self.nblocks
        assert nrecv_peers_block.shape[0] == self.nblocks
        assert reduce_block_idx_block.shape[0] == self.nblocks
        assert reduce_block_cnt_block.shape[0] == self.nblocks
        assert len(reduce_syncer_ptr_arr) == self.nblocks
        assert len(received_arr_block) == self.nblocks
        assert len(reduce_arr_block) == self.nblocks
        assert sm_block_idx_block.shape[0] == self.nblocks
        assert sm_block_cnt_block.shape[0] == self.nblocks
        assert len(sm_syncer_ptr_arr) == self.nblocks
        assert len(reduce_or_get_block) == self.nblocks
        assert skip_signal_block.shape[0] == self.nblocks
        assert len(start_signal_block) == self.nblocks
        assert len(next_start_signal_block) == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_sm_channel_indics.data.ptr) + struct.pack("P", send_sm_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_proxy_channel_indics.data.ptr) + struct.pack("P", send_proxy_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_scratch_arr_mem.data.ptr)
        self.params += struct.pack("P", nrecv_sm_block.data.ptr) + struct.pack("P", nrecv_proxy_block.data.ptr)
        self.params += struct.pack("P", nrecv_peers_block.data.ptr)
        self.params += struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", reduce_block_idx_block.data.ptr) + struct.pack("P", reduce_block_cnt_block.data.ptr)
        self.params += struct.pack("P", reduce_syncer_arr_mem.data.ptr)
        self.params += struct.pack("P", received_arr_mem.data.ptr) + struct.pack("P", reduce_arr_mem.data.ptr)
        self.params += struct.pack("P", sm_block_idx_block.data.ptr) + struct.pack("P", sm_block_cnt_block.data.ptr)
        self.params += struct.pack("P", sm_syncer_arr_mem.data.ptr) + struct.pack("P", reduce_or_get_mem.data.ptr)
        self.params += struct.pack("P", skip_signal_block.data.ptr)
        if "synced" in self.kernel_file:
            self.params += struct.pack("P", start_signal_mem.data.ptr) + struct.pack("P", next_start_signal_mem.data.ptr)

        
        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches,
                      recv_sm_channel_indics, send_sm_channel_indics,
                      recv_proxy_channel_indics, send_proxy_channel_indics,
                      recv_scratch_arr_save, received_arr_save, reduce_arr_save, reduce_or_get_save,
                      recv_scratch_arr_mem, received_arr_mem, reduce_arr_mem, reduce_or_get_mem,
                      start_signal_save, start_signal_mem, next_start_signal_mem,
                      nrecv_sm_block, nrecv_proxy_block, nrecv_peers_block,
                      reduce_block_idx_block, reduce_block_cnt_block,
                      reduce_syncer_arr, reduce_syncer_ptr_arr, reduce_syncer_arr_mem,
                      received_arr_block, reduce_arr_block,
                      sm_block_idx_block, sm_block_cnt_block,
                      sm_syncer_arr, sm_syncer_ptr_arr, sm_syncer_arr_mem,
                      reduce_or_get_block, skip_signal_block,
                      start_signal_block, next_start_signal_block,]
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
            assert all(math.ceil(self.data_chunk_sizes[bid] * nelem_per_chunk / nelem_per_send) <= MAX_NLOOPS for bid in range(self.nblocks))
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None, debug_flag=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        params += struct.pack("Q", debug_flag) if debug_flag is not None else struct.pack("Q", 0)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None, debug_flag=None):
        return self.get_func(nelem_total, nelem_per_send, debug_flag)(stream_ptr)


class ReduceScatterParallelSMHackPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peer of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peer of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        ntrees: int,
        n_parallel_sm_blocks: int = 1,
        leaf_nodes: dict = None,
        skip_leaf_tb: bool = False,
        nthreads=1024,
    ):
        assert (leaf_nodes is not None) == skip_leaf_tb
        n_peers = max([len(recv_sm_channels.get(t, [])) + len(recv_proxy_channels.get(t, [])) for t in range(ntrees)] +
                      [len(send_sm_channels.get(t, [])) + len(send_proxy_channels.get(t, [])) for t in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = REDUCE_SCATTER_PARALLEL_SM_KERNEL_HACK_FILE
        assert "parallel_sm" in self.kernel_file
        self.kernel_name = "pipeline_reduceScatter_hack_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nthreads = nthreads
        self.data = data
    
        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        recv_sm_channel_indics = []
        send_sm_channel_indics = []
        recv_proxy_channel_indics = []
        send_proxy_channel_indics = []
        proxy_recv_scratches_arr = []
        sm_block_idx_arr = []
        sm_block_cnt_arr = []
        sm_syncer_offset = 0
        sm_syncer_indics = []
        skip_signal_arr = []
        self.nblocks = 0
        null_buf = cp.empty(0, dtype=cp.int32)
        assert null_buf.data.ptr == 0
        for tree in range(ntrees):
            assert (tree in recv_sm_channels or tree in send_sm_channels or 
                    tree in recv_proxy_channels or tree in send_proxy_channels)
            assert tree in data_chunk_offsets
            assert tree in data_chunk_sizes
            assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

            if tree in recv_sm_channels:
                assert tree in recv_sm_scratches
                assert len(recv_sm_scratches[tree]) == len(recv_sm_channels[tree])
            if tree in recv_proxy_channels:
                assert tree in recv_proxy_scratches
                assert len(recv_proxy_scratches[tree]) == len(recv_proxy_channels[tree])

            assert len(recv_sm_channels.get(tree, [])) <= 1 and len(recv_proxy_channels.get(tree, [])) <= 1
            recv_sm = len(recv_sm_channels.get(tree, [])) > 0
            recv_proxy = len(recv_proxy_channels.get(tree, [])) > 0
            local_nblocks = n_parallel_sm_blocks if recv_sm else 1
            assert len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) <= 1
            send_sm = len(send_sm_channels.get(tree, [])) > 0
            send_proxy = len(send_proxy_channels.get(tree, [])) > 0
            assert send_sm or send_proxy or recv_sm or recv_proxy

            if leaf_nodes is not None and not recv_sm and not recv_proxy and not send_proxy:
                continue

            self.nblocks += local_nblocks
            recv_sm_channel_indics += [len(recv_sm_handles_arr) if recv_sm else -1] * local_nblocks
            send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1] * local_nblocks
            recv_proxy_channel_indics += [len(recv_proxy_handles_arr) if recv_proxy else -1] * local_nblocks
            send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1] * local_nblocks
            recv_sm_handles_arr += [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
            send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
            recv_proxy_handles_arr += [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
            send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]
            
            if recv_proxy:
                assert len(recv_proxy_scratches.get(tree, [])) == 1
                scratch_buff = recv_proxy_scratches[tree][0]
                proxy_recv_scratches_arr += [struct.pack("P", scratch_buff.data.ptr)] * local_nblocks
            else:
                proxy_recv_scratches_arr += [struct.pack("P", 0)] * local_nblocks
            
            sm_block_idx_arr += list(range(n_parallel_sm_blocks)) if recv_sm else [0]
            sm_block_cnt_arr += [n_parallel_sm_blocks] * n_parallel_sm_blocks if recv_sm else [0]
            if recv_sm:
                sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                sm_syncer_offset += 1
            else:
                sm_syncer_indics += [None]
            
            if recv_sm:
                if leaf_nodes is not None:
                    skip_signal_arr += [leaf_nodes[tree][0]]  * n_parallel_sm_blocks
                else:
                    skip_signal_arr += [False]  * n_parallel_sm_blocks
            else:
                skip_signal_arr += [False]

            self.data_chunk_offsets += [data_chunk_offsets[tree]] * local_nblocks
            self.data_chunk_sizes += [data_chunk_sizes[tree]] * local_nblocks

        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

        recv_sm_handles_mem = cp.asarray(memoryview(b"".join(recv_sm_handles_arr)), dtype=cp.uint8)
        send_sm_handles_mem = cp.asarray(memoryview(b"".join(send_sm_handles_arr)), dtype=cp.uint8)
        recv_proxy_handles_mem = cp.asarray(memoryview(b"".join(recv_proxy_handles_arr)), dtype=cp.uint8)
        send_proxy_handles_mem = cp.asarray(memoryview(b"".join(send_proxy_handles_arr)), dtype=cp.uint8)
        proxy_recv_scratches_mem = cp.asarray(memoryview(b"".join(proxy_recv_scratches_arr)), dtype=cp.uint8)
        assert len(recv_sm_handles_arr) > 0 or recv_sm_handles_mem.data.ptr == 0
        assert len(send_sm_handles_arr) > 0 or send_sm_handles_mem.data.ptr == 0
        assert len(recv_proxy_handles_arr) > 0 or recv_proxy_handles_mem.data.ptr == 0
        assert len(send_proxy_handles_arr) > 0 or send_proxy_handles_mem.data.ptr == 0
        assert len(proxy_recv_scratches_mem) > 0 or proxy_recv_scratches_mem.data.ptr == 0
        recv_sm_channel_indics = cp.array(recv_sm_channel_indics, dtype=cp.int32)
        send_sm_channel_indics = cp.array(send_sm_channel_indics, dtype=cp.int32)
        recv_proxy_channel_indics = cp.array(recv_proxy_channel_indics, dtype=cp.int32)
        send_proxy_channel_indics = cp.array(send_proxy_channel_indics, dtype=cp.int32)
        
        sm_block_idx_arr = cp.array(sm_block_idx_arr, dtype=cp.int32)
        sm_block_cnt_arr = cp.array(sm_block_cnt_arr, dtype=cp.int32)

        sm_syncer_num = sm_syncer_offset
        sm_syncer_arr = cp.zeros(sm_syncer_num * 12, dtype=cp.bool_)
        sm_syncer_ptr_arr = [struct.pack("P", sm_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in sm_syncer_indics]
        sm_syncer_arr_mem = cp.asarray(memoryview(b"".join(sm_syncer_ptr_arr)), dtype=cp.uint8)
        skip_signal_arr = cp.array(skip_signal_arr, dtype=cp.bool_)

        if leaf_nodes is None:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        else:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels
            assert len(send_sm_handles_arr) == sum(0 if len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) == 0
                                                   else len(send_sm_channels.get(tree, [])) for tree in range(ntrees))
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(proxy_recv_scratches_arr) == self.nblocks
        assert recv_sm_channel_indics.shape[0] == self.nblocks
        assert send_sm_channel_indics.shape[0] == self.nblocks
        assert recv_proxy_channel_indics.shape[0] == self.nblocks
        assert send_proxy_channel_indics.shape[0] == self.nblocks
        assert sm_block_idx_arr.shape[0] == self.nblocks
        assert sm_block_cnt_arr.shape[0] == self.nblocks
        assert len(sm_syncer_ptr_arr) == self.nblocks
        assert skip_signal_arr.shape[0] == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_sm_channel_indics.data.ptr) + struct.pack("P", send_sm_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_proxy_channel_indics.data.ptr) + struct.pack("P", send_proxy_channel_indics.data.ptr)
        self.params += struct.pack("P", proxy_recv_scratches_mem.data.ptr) + struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", sm_block_idx_arr.data.ptr) + struct.pack("P", sm_block_cnt_arr.data.ptr)
        self.params += struct.pack("P", sm_syncer_arr_mem.data.ptr) + struct.pack("P", skip_signal_arr.data.ptr)

        
        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches,
                      proxy_recv_scratches_mem, proxy_recv_scratches_arr,
                      recv_sm_channel_indics, send_sm_channel_indics,
                      recv_proxy_channel_indics, send_proxy_channel_indics,
                      sm_block_idx_arr, sm_block_cnt_arr, skip_signal_arr,
                      sm_syncer_arr, sm_syncer_ptr_arr, sm_syncer_arr_mem]
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
            assert all(math.ceil(self.data_chunk_sizes[bid] * nelem_per_chunk / nelem_per_send) <= MAX_NLOOPS for bid in range(self.nblocks))
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None, debug_flag=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        params += struct.pack("Q", debug_flag) if debug_flag is not None else struct.pack("Q", 0)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None, debug_flag=None):
        return self.get_func(nelem_total, nelem_per_send, debug_flag)(stream_ptr)


def verify_spanning_tree(G: nx.DiGraph, nranks: int, root: int):
    assert G.number_of_nodes() == nranks
    assert G.number_of_edges() == nranks - 1
    assert nx.is_weakly_connected(G)
    for v in G.nodes():
        assert G.in_degree(v) == (1 if v != root else 0)


class ReduceScatterParallelSMSendTBPipelineKernel:
    def __init__(
        self,
        recv_sm_channels: dict,  # recv_sm_channels[tree] = sm recv peers of tree
        send_sm_channels: dict,  # send_sm_channels[tree] = sm send peer of tree
        recv_proxy_channels: dict,  # recv_proxy_channels[tree] = proxy recv peers of tree
        send_proxy_channels: dict,  # send_proxy_channels[tree] = proxy send peer of tree
        data: cp.ndarray,
        data_chunk_offsets: dict,   # data_chunk_offsets[tree] = chunk offset of tree
        data_chunk_sizes: dict,     # data_chunk_sizes[tree] = data nchunks of tree
        total_chunks: int,
        scratch_size: int,
        recv_sm_scratches: dict,
        recv_proxy_scratches: dict,
        ntrees: int,
        n_parallel_sm_blocks: int = 1,
        n_parallel_reduce_blocks: int = 1,
        leaf_nodes: dict = None,
        skip_leaf_tb: bool = False,
        nthreads=1024,
    ):
        assert (leaf_nodes is not None) == skip_leaf_tb
        n_peers = max([len(recv_sm_channels.get(t, [])) + len(recv_proxy_channels.get(t, [])) for t in range(ntrees)] +
                      [len(send_sm_channels.get(t, [])) + len(send_proxy_channels.get(t, [])) for t in range(ntrees)] + [1])
        assert n_peers <= 8, "N_PEERS=8 in pipeline_kernel.cu"
        n_recv_sm_channels = sum(len(l) for l in recv_sm_channels.values())
        n_send_sm_channels = sum(len(l) for l in send_sm_channels.values())
        n_recv_proxy_channels = sum(len(l) for l in recv_proxy_channels.values())
        n_send_proxy_channels = sum(len(l) for l in send_proxy_channels.values())
        assert n_recv_proxy_channels + n_send_proxy_channels <= 128, "see https://github.com/microsoft/mscclpp/issues/242"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.kernel_file = REDUCE_SCATTER_PARALLEL_SM_KERNEL_SENDTB_FILE
        assert "parallel_sm" in self.kernel_file
        self.kernel_name = "pipeline_reduceScatter_sendtb_schedule"
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        self.nthreads = nthreads
        self.data = data
    
        self.data_chunk_offsets = []
        self.data_chunk_sizes = []
        self.total_chunks = total_chunks
        self.scratch_size = scratch_size
        self.use_schatch = len(recv_sm_scratches) > 0 or len(recv_proxy_scratches) > 0

        recv_scratch_ptr_arrs = []
        pending_receives_cnt_arrs = []
        sent_progresses = []
        reduce_peer_ptr_arr = []

        recv_sm_handles_arr = []
        send_sm_handles_arr = []
        recv_proxy_handles_arr = []
        send_proxy_handles_arr = []
        recv_sm_channel_indics = []
        send_sm_channel_indics = []
        recv_proxy_channel_indics = []
        send_proxy_channel_indics = []
        threadblock_type_arr = []
        recv_scratch_arr_arr = []
        recv_scratch_arr = []
        pending_receives_arr_arr = []
        pending_receives_arr = []
        nrecv_peers_arr = []
        nrecv_sm_arr = []
        nrecv_proxy_arr = []
        sent_progress_arr = []
        sm_block_idx_arr = []
        sm_block_cnt_arr = []
        sm_syncer_offset = 0
        sm_syncer_indics = []
        reduce_block_idx_arr = []
        reduce_block_cnt_arr = []
        reduce_syncer_offset = 0
        reduce_syncer_indics = []
        reduce_peer_arr = []
        skip_signal_arr = []
        self.nblocks = 0
        null_buf = cp.empty(0, dtype=cp.int32)
        assert null_buf.data.ptr == 0
        for tree in range(ntrees):
            assert (tree in recv_sm_channels or tree in send_sm_channels or 
                    tree in recv_proxy_channels or tree in send_proxy_channels)
            assert tree in data_chunk_offsets
            assert tree in data_chunk_sizes
            assert data_chunk_offsets[tree] + data_chunk_sizes[tree] <= total_chunks

            if tree in recv_sm_channels:
                assert tree in recv_sm_scratches
                assert len(recv_sm_scratches[tree]) == len(recv_sm_channels[tree])
            if tree in recv_proxy_channels:
                assert tree in recv_proxy_scratches
                assert len(recv_proxy_scratches[tree]) == len(recv_proxy_channels[tree])

            nrecv_sm = len(recv_sm_channels.get(tree, []))
            nrecv_proxy = len(recv_proxy_channels.get(tree, []))
            nrecv_peers = nrecv_sm + nrecv_proxy
            assert nrecv_peers <= n_peers
            assert len(send_sm_channels.get(tree, [])) + len(send_proxy_channels.get(tree, [])) <= 1
            send_sm = len(send_sm_channels.get(tree, [])) > 0
            send_proxy = len(send_proxy_channels.get(tree, [])) > 0
            assert send_sm or send_proxy or nrecv_peers > 0

            if nrecv_sm == 1 and nrecv_proxy == 0:
                self.nblocks += n_parallel_sm_blocks
                recv_sm_channel_indics += [len(recv_sm_handles_arr)] * n_parallel_sm_blocks
                if leaf_nodes is not None:
                    skip_signal_arr += [leaf_nodes[tree][0]] * n_parallel_sm_blocks
                else:
                    skip_signal_arr += [False] * n_parallel_sm_blocks
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1] * n_parallel_sm_blocks
                recv_proxy_channel_indics += [-1] * n_parallel_sm_blocks
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1] * n_parallel_sm_blocks

                threadblock_type_arr += [-1] * n_parallel_sm_blocks
                recv_scratch_arr_arr += [struct.pack("P", 0)] * n_parallel_sm_blocks
                assert len(recv_sm_scratches.get(tree, [])) == 1
                assert len(recv_proxy_scratches.get(tree, [])) == 0
                recv_scratch_arr += [struct.pack("P", recv_sm_scratches[tree][0].data.ptr)] * n_parallel_sm_blocks

                pending_receives_arr_arr += [struct.pack("P", 0)] * n_parallel_sm_blocks
                pending_receives_arr += [struct.pack("P", 0)] * n_parallel_sm_blocks
                nrecv_peers_arr += [nrecv_peers] * n_parallel_sm_blocks
                nrecv_sm_arr += [nrecv_sm] * n_parallel_sm_blocks
                nrecv_proxy_arr += [nrecv_proxy] * n_parallel_sm_blocks
                sent_progress_arr += [struct.pack("P", 0)] * n_parallel_sm_blocks

                sm_block_idx_arr += list(range(n_parallel_sm_blocks))
                sm_block_cnt_arr += [n_parallel_sm_blocks] * n_parallel_sm_blocks
                sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                sm_syncer_offset += 1

                reduce_block_idx_arr += [-1] * n_parallel_sm_blocks
                reduce_block_cnt_arr += [0] * n_parallel_sm_blocks
                reduce_syncer_indics += [None] * n_parallel_sm_blocks
                reduce_peer_arr += [struct.pack("P", 0)] * n_parallel_sm_blocks

                self.data_chunk_offsets += [data_chunk_offsets[tree]] * n_parallel_sm_blocks
                self.data_chunk_sizes += [data_chunk_sizes[tree]] * n_parallel_sm_blocks
            elif nrecv_peers == 0:
                if leaf_nodes is not None and not send_proxy:
                    continue

                self.nblocks += 1
                recv_sm_channel_indics += [-1]
                skip_signal_arr += [False]
                send_sm_channel_indics += [len(send_sm_handles_arr) if send_sm else -1]
                recv_proxy_channel_indics += [-1]
                send_proxy_channel_indics += [len(send_proxy_handles_arr) if send_proxy else -1]

                threadblock_type_arr += [1]
                recv_scratch_arr_arr += [struct.pack("P", 0)]
                assert len(recv_sm_scratches.get(tree, [])) == 0
                assert len(recv_proxy_scratches.get(tree, [])) == 0
                recv_scratch_arr += [struct.pack("P", 0)]

                pending_receives_arr_arr += [struct.pack("P", 0)]
                pending_receives_arr += [struct.pack("P", 0)]
                nrecv_peers_arr += [nrecv_peers]
                nrecv_sm_arr += [nrecv_sm]
                nrecv_proxy_arr += [nrecv_proxy]
                sent_progress_arr += [struct.pack("P", 0)]

                sm_block_idx_arr += [-1]
                sm_block_cnt_arr += [0]
                sm_syncer_indics += [None]

                reduce_block_idx_arr += [0]
                reduce_block_cnt_arr += [1]
                reduce_syncer_indics += [-1]
                reduce_peer_arr += [struct.pack("P", 0)]

                self.data_chunk_offsets += [data_chunk_offsets[tree]]
                self.data_chunk_sizes += [data_chunk_sizes[tree]]
            else:
                local_nblocks = nrecv_sm * n_parallel_sm_blocks + n_parallel_reduce_blocks
                self.nblocks += local_nblocks

                for i in range(nrecv_sm):
                    recv_sm_channel_indics += [len(recv_sm_handles_arr) + i] * n_parallel_sm_blocks
                    if leaf_nodes is not None:
                        skip_signal_arr += [leaf_nodes[tree][i]]  * n_parallel_sm_blocks
                    else:
                        skip_signal_arr += [False] * n_parallel_sm_blocks
                    threadblock_type_arr += [-1] * n_parallel_sm_blocks
                recv_sm_channel_indics += [-1] * n_parallel_reduce_blocks
                skip_signal_arr += [False] * n_parallel_reduce_blocks
                threadblock_type_arr += [1] * n_parallel_reduce_blocks
                send_sm_channel_indics += [-1] * nrecv_sm * n_parallel_sm_blocks + [len(send_sm_handles_arr) if send_sm else -1] * n_parallel_reduce_blocks
                recv_proxy_channel_indics += [-1] * nrecv_sm * n_parallel_sm_blocks + [len(recv_proxy_handles_arr)] * n_parallel_reduce_blocks
                send_proxy_channel_indics += [-1] * nrecv_sm * n_parallel_sm_blocks + [len(send_proxy_handles_arr) if send_proxy else -1] * n_parallel_reduce_blocks

                recv_scratch_ptr_arr = [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_sm_scratches.get(tree, [])] + \
                                       [struct.pack("P", scratch_buff.data.ptr) for scratch_buff in recv_proxy_scratches.get(tree, [])]
                recv_scratch_ptr_mem = cp.asarray(memoryview(b"".join(recv_scratch_ptr_arr)), dtype=cp.uint8)
                recv_scratch_ptr_arrs.append((recv_scratch_ptr_arr, recv_scratch_ptr_mem))

                recv_scratch_arr_arr += [struct.pack("P", 0)] * nrecv_sm * n_parallel_sm_blocks + [struct.pack("P", recv_scratch_ptr_mem.data.ptr)] * n_parallel_reduce_blocks
                assert len(recv_sm_scratches.get(tree, [])) == nrecv_sm
                assert len(recv_proxy_scratches.get(tree, [])) == nrecv_proxy
                for scratch_buff in recv_sm_scratches.get(tree, []):
                    recv_scratch_arr += [struct.pack("P", scratch_buff.data.ptr)] * n_parallel_sm_blocks
                recv_scratch_arr += [struct.pack("P", 0)] * n_parallel_reduce_blocks

                pending_receives_cnt_arr = cp.zeros(nrecv_sm, dtype=cp.int32)
                pending_receives_cnt_arrs.append(pending_receives_cnt_arr)

                pending_receives_arr_arr += [struct.pack("P", 0)] * nrecv_sm * n_parallel_sm_blocks + [struct.pack("P", pending_receives_cnt_arr.data.ptr)] * n_parallel_reduce_blocks
                for i in range(nrecv_sm):
                    pending_receives_arr += [struct.pack("P", pending_receives_cnt_arr.data.ptr + i * 4)] * n_parallel_sm_blocks
                pending_receives_arr += [struct.pack("P", 0)] * n_parallel_reduce_blocks
                nrecv_peers_arr += [nrecv_peers] * local_nblocks
                nrecv_sm_arr += [nrecv_sm] * local_nblocks
                nrecv_proxy_arr += [nrecv_proxy] * local_nblocks
                
                sent_progress = cp.zeros(1, dtype=cp.int32)
                sent_progresses.append(sent_progress)
                sent_progress_arr += [struct.pack("P", sent_progress.data.ptr)] * local_nblocks

                sm_block_idx_arr += list(range(n_parallel_sm_blocks)) * nrecv_sm + [-1] * n_parallel_reduce_blocks
                sm_block_cnt_arr += [n_parallel_sm_blocks] * nrecv_sm * n_parallel_sm_blocks + [0] * n_parallel_reduce_blocks
                for i in range(nrecv_sm):
                    sm_syncer_indics += [sm_syncer_offset] * n_parallel_sm_blocks
                    sm_syncer_offset += 1
                sm_syncer_indics += [None] * n_parallel_reduce_blocks

                reduce_block_idx_arr += [-1] * nrecv_sm * n_parallel_sm_blocks + list(range(n_parallel_reduce_blocks))
                reduce_block_cnt_arr += [0] * nrecv_sm * n_parallel_sm_blocks + [n_parallel_reduce_blocks] * n_parallel_reduce_blocks
                reduce_syncer_indics += [None] * nrecv_sm * n_parallel_sm_blocks + [reduce_syncer_offset] * n_parallel_reduce_blocks
                reduce_syncer_offset += 1

                reduce_peer = cp.zeros(1, dtype=cp.int32)
                reduce_peer_ptr_arr.append(reduce_peer)
                reduce_peer_arr += [struct.pack("P", 0)] * nrecv_sm * n_parallel_sm_blocks + [struct.pack("P", reduce_peer.data.ptr)] * n_parallel_reduce_blocks

                self.data_chunk_offsets += [data_chunk_offsets[tree]] * local_nblocks
                self.data_chunk_sizes += [data_chunk_sizes[tree]] * local_nblocks

            recv_sm_handles_arr += [ch.device_handle().raw for ch in recv_sm_channels.get(tree, [])]
            send_sm_handles_arr += [ch.device_handle().raw for ch in send_sm_channels.get(tree, [])]
            recv_proxy_handles_arr += [ch.device_handle().raw for ch in recv_proxy_channels.get(tree, [])]
            send_proxy_handles_arr += [ch.device_handle().raw for ch in send_proxy_channels.get(tree, [])]

        if self.nblocks > MAX_NBLOCKS:
            raise ThreadBlockLimitException(f"nblocks={self.nblocks} > MAX_NBLOCKS", self.nblocks)
        if self.nblocks > 100:
            print(f"Warning: nblocks={self.nblocks} > 100", flush=True)

        recv_sm_handles_mem = cp.asarray(memoryview(b"".join(recv_sm_handles_arr)), dtype=cp.uint8)
        send_sm_handles_mem = cp.asarray(memoryview(b"".join(send_sm_handles_arr)), dtype=cp.uint8)
        recv_proxy_handles_mem = cp.asarray(memoryview(b"".join(recv_proxy_handles_arr)), dtype=cp.uint8)
        send_proxy_handles_mem = cp.asarray(memoryview(b"".join(send_proxy_handles_arr)), dtype=cp.uint8)
        recv_scratch_mem = cp.asarray(memoryview(b"".join(recv_scratch_arr)), dtype=cp.uint8)
        assert len(recv_sm_handles_arr) > 0 or recv_sm_handles_mem.data.ptr == 0
        assert len(send_sm_handles_arr) > 0 or send_sm_handles_mem.data.ptr == 0
        assert len(recv_proxy_handles_arr) > 0 or recv_proxy_handles_mem.data.ptr == 0
        assert len(send_proxy_handles_arr) > 0 or send_proxy_handles_mem.data.ptr == 0
        assert len(recv_scratch_arr) > 0 or recv_scratch_mem.data.ptr == 0
        recv_sm_channel_indics = cp.array(recv_sm_channel_indics, dtype=cp.int32)
        send_sm_channel_indics = cp.array(send_sm_channel_indics, dtype=cp.int32)
        recv_proxy_channel_indics = cp.array(recv_proxy_channel_indics, dtype=cp.int32)
        send_proxy_channel_indics = cp.array(send_proxy_channel_indics, dtype=cp.int32)
        threadblock_type_arr = cp.array(threadblock_type_arr, dtype=cp.int32)
        recv_scratch_arr_mem = cp.asarray(memoryview(b"".join(recv_scratch_arr_arr)), dtype=cp.uint8)
        pending_receives_arr_mem = cp.asarray(memoryview(b"".join(pending_receives_arr_arr)), dtype=cp.uint8)
        pending_receives_mem = cp.asarray(memoryview(b"".join(pending_receives_arr)), dtype=cp.uint8)
        nrecv_peers_arr = cp.array(nrecv_peers_arr, dtype=cp.int32)
        nrecv_sm_arr = cp.array(nrecv_sm_arr, dtype=cp.int32)
        nrecv_proxy_arr = cp.array(nrecv_proxy_arr, dtype=cp.int32)
        sent_progress_mem = cp.asarray(memoryview(b"".join(sent_progress_arr)), dtype=cp.uint8)
        reduce_peer_mem = cp.asarray(memoryview(b"".join(reduce_peer_arr)), dtype=cp.uint8)
        
        sm_block_idx_arr = cp.array(sm_block_idx_arr, dtype=cp.int32)
        sm_block_cnt_arr = cp.array(sm_block_cnt_arr, dtype=cp.int32)

        sm_syncer_num = sm_syncer_offset
        sm_syncer_arr = cp.zeros(sm_syncer_num * 12, dtype=cp.bool_)
        sm_syncer_ptr_arr = [struct.pack("P", sm_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in sm_syncer_indics]
        sm_syncer_arr_mem = cp.asarray(memoryview(b"".join(sm_syncer_ptr_arr)), dtype=cp.uint8)
        skip_signal_arr = cp.array(skip_signal_arr, dtype=cp.bool_)

        reduce_block_idx_arr = cp.array(reduce_block_idx_arr, dtype=cp.int32)
        reduce_block_cnt_arr = cp.array(reduce_block_cnt_arr, dtype=cp.int32)

        reduce_syncer_num = reduce_syncer_offset
        reduce_syncer_arr = cp.zeros(reduce_syncer_num * 12, dtype=cp.bool_)
        reduce_syncer_ptr_arr = [struct.pack("P", reduce_syncer_arr.data.ptr + i * 12) if i is not None else struct.pack("P", 0) for i in reduce_syncer_indics]
        reduce_syncer_arr_mem = cp.asarray(memoryview(b"".join(reduce_syncer_ptr_arr)), dtype=cp.uint8)

        if leaf_nodes is None:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels and len(send_sm_handles_arr) == n_send_sm_channels
        else:
            assert len(recv_sm_handles_arr) == n_recv_sm_channels
            assert len(send_sm_handles_arr) == sum(0 if len(recv_sm_channels.get(tree, [])) + len(recv_proxy_channels.get(tree, [])) == 0
                                                   else len(send_sm_channels.get(tree, [])) for tree in range(ntrees))
        assert len(recv_proxy_handles_arr) == n_recv_proxy_channels and len(send_proxy_handles_arr) == n_send_proxy_channels
        assert len(recv_scratch_arr) == self.nblocks
        assert recv_sm_channel_indics.shape[0] == self.nblocks
        assert send_sm_channel_indics.shape[0] == self.nblocks
        assert recv_proxy_channel_indics.shape[0] == self.nblocks
        assert send_proxy_channel_indics.shape[0] == self.nblocks
        assert threadblock_type_arr.shape[0] == self.nblocks
        assert len(recv_scratch_arr_arr) == self.nblocks
        assert len(pending_receives_arr_arr) == self.nblocks
        assert len(pending_receives_arr) == self.nblocks
        assert nrecv_peers_arr.shape[0] == self.nblocks
        assert nrecv_sm_arr.shape[0] == self.nblocks
        assert nrecv_proxy_arr.shape[0] == self.nblocks
        assert len(sent_progress_arr) == self.nblocks
        assert sm_block_idx_arr.shape[0] == self.nblocks
        assert sm_block_cnt_arr.shape[0] == self.nblocks
        assert len(sm_syncer_ptr_arr) == self.nblocks
        assert reduce_block_idx_arr.shape[0] == self.nblocks
        assert reduce_block_cnt_arr.shape[0] == self.nblocks
        assert len(reduce_syncer_ptr_arr) == self.nblocks
        assert skip_signal_arr.shape[0] == self.nblocks
        assert len(self.data_chunk_offsets) == self.nblocks
        assert len(self.data_chunk_sizes) == self.nblocks

        self.params = b""
        self.params += struct.pack("P", recv_sm_handles_mem.data.ptr) + struct.pack("P", send_sm_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_proxy_handles_mem.data.ptr) + struct.pack("P", send_proxy_handles_mem.data.ptr)
        self.params += struct.pack("P", recv_sm_channel_indics.data.ptr) + struct.pack("P", send_sm_channel_indics.data.ptr)
        self.params += struct.pack("P", recv_proxy_channel_indics.data.ptr) + struct.pack("P", send_proxy_channel_indics.data.ptr)
        self.params += struct.pack("P", threadblock_type_arr.data.ptr)
        self.params += struct.pack("P", recv_scratch_arr_mem.data.ptr) + struct.pack("P", recv_scratch_mem.data.ptr)
        self.params += struct.pack("Q", scratch_size) + struct.pack("P", data.data.ptr)
        self.params += struct.pack("P", pending_receives_arr_mem.data.ptr) + struct.pack("P", pending_receives_mem.data.ptr)
        self.params += struct.pack("P", nrecv_peers_arr.data.ptr) + struct.pack("P", nrecv_sm_arr.data.ptr) + struct.pack("P", nrecv_proxy_arr.data.ptr)
        self.params += struct.pack("P", sent_progress_mem.data.ptr)
        self.params += struct.pack("P", sm_block_idx_arr.data.ptr) + struct.pack("P", sm_block_cnt_arr.data.ptr)
        self.params += struct.pack("P", sm_syncer_arr_mem.data.ptr) + struct.pack("P", skip_signal_arr.data.ptr)
        self.params += struct.pack("P", reduce_peer_mem.data.ptr)
        self.params += struct.pack("P", reduce_block_idx_arr.data.ptr) + struct.pack("P", reduce_block_cnt_arr.data.ptr)
        self.params += struct.pack("P", reduce_syncer_arr_mem.data.ptr)
        
        # keep references to avoid garbage collection
        self._temp = [recv_sm_channels, send_sm_channels,
                      recv_proxy_channels, send_proxy_channels,
                      recv_sm_handles_mem, send_sm_handles_mem,
                      recv_proxy_handles_mem, send_proxy_handles_mem,
                      data, recv_sm_scratches, recv_proxy_scratches, recv_scratch_mem,
                      recv_scratch_arr_mem, pending_receives_arr_mem, pending_receives_mem,
                      sent_progress_mem, reduce_peer_mem, reduce_peer_ptr_arr,
                      recv_scratch_ptr_arrs, pending_receives_cnt_arrs, sent_progresses,
                      recv_sm_channel_indics, send_sm_channel_indics,
                      recv_proxy_channel_indics, send_proxy_channel_indics,
                      threadblock_type_arr, recv_scratch_arr_arr, recv_scratch_arr,
                      pending_receives_arr_arr, pending_receives_arr,
                      nrecv_peers_arr, nrecv_sm_arr, nrecv_proxy_arr, sent_progress_arr,
                      sm_block_idx_arr, sm_block_cnt_arr, skip_signal_arr,
                      sm_syncer_arr, sm_syncer_ptr_arr, sm_syncer_arr_mem,
                      reduce_block_idx_arr, reduce_block_cnt_arr,
                      reduce_syncer_arr, reduce_syncer_ptr_arr, reduce_syncer_arr_mem]
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
            assert all(math.ceil(self.data_chunk_sizes[bid] * nelem_per_chunk / nelem_per_send) <= MAX_NLOOPS for bid in range(self.nblocks))
            nelem_totals = cp.array([self.data_chunk_sizes[bid] * nelem_per_chunk for bid in range(self.nblocks)], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (data_starts, nelem_totals)

        params = self.params + struct.pack("P", data_starts.data.ptr) + struct.pack("Q", nelem_per_send) + struct.pack("P", nelem_totals.data.ptr)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nelem_total=None, nelem_per_send=None, debug_flag=None):
        if nelem_per_send is None:
            nelem_per_send = self.scratch_size
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total, nelem_per_send)
        params += struct.pack("Q", debug_flag) if debug_flag is not None else struct.pack("Q", 0)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, stream_ptr)


    def __call__(self, nelem_total=None, nelem_per_send=None, stream_ptr=None, debug_flag=None):
        return self.get_func(nelem_total, nelem_per_send, debug_flag)(stream_ptr)


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
                  scratch_size: int, leaf_nodes: dict = None):
    recv_peers, send_peers = list(recv_peers), list(send_peers)
    recv_sm_channels, recv_proxy_channels = [], []
    recv_sm_scratches, recv_proxy_scratches = ([], []) if scratch_size is not None else (None, None)
    recv_leaf_nodes = []
    for dest in recv_peers:
        connect = connections[dest]
        recv_buf = cp.empty(scratch_size, dtype=cp.int32) if scratch_size is not None else data
        if connection_types[dest] == "sm":
            assert connect.transport() == Transport.CudaIpc
            if scratch_size is not None:
                recv_sm_scratches.append(recv_buf)
            recv_sm_channels.append(group.make_sm_channel(recv_buf, connect, dest))
            if leaf_nodes is not None:
                recv_leaf_nodes.append(leaf_nodes[dest])
            else:
                recv_leaf_nodes.append(False)
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
        assert len(recv_sm_channels) == len(recv_leaf_nodes)
        return (recv_sm_channels, send_sm_channels,
                recv_proxy_channels, send_proxy_channels,
                recv_sm_scratches, recv_proxy_scratches, recv_leaf_nodes)
    else:
        assert leaf_nodes is None, "allgather cannot skip leaf nodes"
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
             recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id], _) = \
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
             recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id], _) = \
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
                     proxy_service: ProxyService = None, n_parallel_sm_blocks: int = None):
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

    if n_parallel_sm_blocks is not None:
        args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                    recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                    data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                    total_chunks=total_chunks, node_types=node_types, ntrees=nblocks,
                    n_parallel_sm_blocks=n_parallel_sm_blocks)
        kernel = AllgatherParallelSMPipelineKernel(**args)
    else:
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
                          scratch_size: int, proxy_service: ProxyService = None,
                          use_reduceScatter_kernel=False, n_parallel_sm_blocks: int = None,
                          n_parallel_reduce_blocks: int = None, coll_re: bool = False,
                          skip_leaf_tb: bool = False, sendtb: bool = False,
                          n_pipeline: int = None):
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
    if n_pipeline is None:
        n_pipeline = 1
    nchunks_per_shard = k * n_pipeline
    total_chunks = nchunks_per_shard * group.nranks

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    recv_sm_scratches, recv_proxy_scratches = {}, {}
    node_types = {}
    data_chunk_offsets = {}
    data_chunk_sizes = {}
    nblocks = 0
    leaf_nodes = {}
    
    pipeline_groups = {}
    for pp in range(n_pipeline):
        for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
            test_G = nx.DiGraph()
            test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
            verify_spanning_tree(test_G, group.nranks, u)

            chunk = u * nchunks_per_shard + chunk_starts[u, i] * n_pipeline + pp * Cs[u, i]

            children = list(test_G.successors(group.my_rank))
            if skip_leaf_tb:
                children_leaf_nodes = {child: test_G.out_degree(child) == 0 for child in children}
            else:
                children_leaf_nodes = None

            # reduce node
            tb_id = nblocks
            pipeline_groups[u, i] = pipeline_groups.get((u, i), []) + [tb_id]
            nblocks += 1
            (recv_sm_channels[tb_id], send_sm_channels[tb_id],
                recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
                recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id], leaf_nodes[tb_id]) = \
                make_channels(group=group, proxy_service=proxy_service, connections=connections,
                                connection_types=connection_types, recv_peers=children,
                                send_peers=list(test_G.predecessors(group.my_rank)),
                                data=data, scratch_size=scratch_size,
                                leaf_nodes=children_leaf_nodes)
            assert len(send_sm_channels[tb_id]) + len(send_proxy_channels[tb_id]) <= 1
            node_types[tb_id] = -1
            data_chunk_offsets[tb_id] = chunk
            data_chunk_sizes[tb_id] = Cs[u, i]
    pipeline_groups = [grp for grp in pipeline_groups.values()]
    
    assert not (use_reduceScatter_kernel and n_parallel_sm_blocks is not None)
    assert not (use_reduceScatter_kernel and sendtb)
    if use_reduceScatter_kernel:
        assert n_parallel_sm_blocks is None
        assert n_parallel_reduce_blocks is None
        assert not skip_leaf_tb
        assert not coll_re
        assert n_pipeline is None
        for tree in range(nblocks):
            if len(recv_sm_channels[tree]) == 0:
                assert len(recv_sm_scratches[tree]) == 0
                del recv_sm_channels[tree], recv_sm_scratches[tree]
            if len(recv_proxy_channels[tree]) == 0:
                assert len(recv_proxy_scratches[tree]) == 0
                del recv_proxy_channels[tree], recv_proxy_scratches[tree]
        args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                ntrees=nblocks)
        kernel = ReduceScatterPipelineKernel(**args)
    elif sendtb:
        assert n_parallel_sm_blocks is not None
        assert n_parallel_reduce_blocks is not None
        assert not coll_re
        assert n_pipeline is None
        for tree in range(nblocks):
            if len(recv_sm_channels[tree]) == 0:
                assert len(recv_sm_scratches[tree]) == 0
                del recv_sm_channels[tree], recv_sm_scratches[tree]
            if len(recv_proxy_channels[tree]) == 0:
                assert len(recv_proxy_scratches[tree]) == 0
                del recv_proxy_channels[tree], recv_proxy_scratches[tree]
        args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                ntrees=nblocks, n_parallel_sm_blocks=n_parallel_sm_blocks,
                n_parallel_reduce_blocks=n_parallel_reduce_blocks,
                leaf_nodes=leaf_nodes if skip_leaf_tb else None, skip_leaf_tb=skip_leaf_tb)
        kernel = ReduceScatterParallelSMSendTBPipelineKernel(**args)
    elif n_parallel_sm_blocks is not None:
        assert n_parallel_sm_blocks is not None
        assert n_parallel_reduce_blocks is None or coll_re
        for tree in range(nblocks):
            if len(recv_sm_channels[tree]) == 0:
                assert len(recv_sm_scratches[tree]) == 0
                del recv_sm_channels[tree], recv_sm_scratches[tree]
            if len(recv_proxy_channels[tree]) == 0:
                assert len(recv_proxy_scratches[tree]) == 0
                del recv_proxy_channels[tree], recv_proxy_scratches[tree]
        args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                ntrees=nblocks, n_parallel_sm_blocks=n_parallel_sm_blocks,
                leaf_nodes=leaf_nodes if skip_leaf_tb else None, skip_leaf_tb=skip_leaf_tb)
        if coll_re:
            args["n_parallel_reduce_blocks"] = n_parallel_reduce_blocks
            args["pipeline_groups"] = pipeline_groups
            kernel = ReduceScatterParallelSMCollREPipelineKernel(**args)
        else:
            assert n_pipeline is None
            kernel = ReduceScatterParallelSMPipelineKernel(**args)
    else:
        assert n_parallel_sm_blocks is None
        assert n_parallel_reduce_blocks is None
        assert not skip_leaf_tb
        assert not coll_re
        assert n_pipeline is None
        args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
                recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
                data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
                total_chunks=total_chunks, scratch_size=scratch_size,
                recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
                node_types=node_types, nblocks=nblocks)
        kernel = PipelineKernel(**args)

    return kernel


def reduce_scatter_kernel_hack(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                               connections: dict, connection_types: dict, data: cp.ndarray,
                               scratch_size: int, proxy_service: ProxyService = None,
                               n_parallel_sm_blocks: int = None, skip_leaf_tb: bool = False):
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
    leaf_nodes = {}

    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        test_G = nx.DiGraph()
        test_G.add_edges_from((p[0][0], p[-1][-1]) for p in ps)
        verify_spanning_tree(test_G, group.nranks, u)

        chunk = u * nchunks_per_shard + chunk_starts[u, i]

        children = list(test_G.successors(group.my_rank))
        if skip_leaf_tb:
            children_leaf_nodes = {child: test_G.out_degree(child) == 0 for child in children}
        else:
            children_leaf_nodes = None

        # reduce node
        tb_id = nblocks
        nblocks += 1
        (recv_sm_channels[tb_id], send_sm_channels[tb_id],
            recv_proxy_channels[tb_id], send_proxy_channels[tb_id],
            recv_sm_scratches[tb_id], recv_proxy_scratches[tb_id], leaf_nodes[tb_id]) = \
            make_channels(group=group, proxy_service=proxy_service, connections=connections,
                            connection_types=connection_types, recv_peers=children,
                            send_peers=list(test_G.predecessors(group.my_rank)),
                            data=data, scratch_size=scratch_size,
                            leaf_nodes=children_leaf_nodes)
        assert len(send_sm_channels[tb_id]) + len(send_proxy_channels[tb_id]) <= 1
        assert len(recv_sm_channels[tb_id]) <= 1 and len(recv_proxy_channels[tb_id]) <= 1
        node_types[tb_id] = -1
        data_chunk_offsets[tb_id] = chunk
        data_chunk_sizes[tb_id] = Cs[u, i]
    
    for tree in range(nblocks):
        if len(recv_sm_channels[tree]) == 0:
            assert len(recv_sm_scratches[tree]) == 0
            del recv_sm_channels[tree], recv_sm_scratches[tree]
        if len(recv_proxy_channels[tree]) == 0:
            assert len(recv_proxy_scratches[tree]) == 0
            del recv_proxy_channels[tree], recv_proxy_scratches[tree]
    args = dict(recv_sm_channels=recv_sm_channels, send_sm_channels=send_sm_channels,
            recv_proxy_channels=recv_proxy_channels, send_proxy_channels=send_proxy_channels,
            data=data, data_chunk_offsets=data_chunk_offsets, data_chunk_sizes=data_chunk_sizes,
            total_chunks=total_chunks, scratch_size=scratch_size,
            recv_sm_scratches=recv_sm_scratches, recv_proxy_scratches=recv_proxy_scratches,
            ntrees=nblocks, n_parallel_sm_blocks=n_parallel_sm_blocks,
            leaf_nodes=leaf_nodes if skip_leaf_tb else None, skip_leaf_tb=skip_leaf_tb)
    kernel = ReduceScatterParallelSMHackPipelineKernel(**args)

    return kernel
