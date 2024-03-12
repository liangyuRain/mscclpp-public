import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np
from mpi4py import MPI
import struct
import uuid
import os

from mscclpp import (
    ProxyService,
    Transport,
)
import mscclpp.comm as mscclpp_comm
from .mscclpp_mpi import MpiGroup
from mscclpp import ProxyService
from mscclpp.utils import KernelBuilder


def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream)
        graph = stream.end_capture()

    # now run a warm up round
    graph.launch(stream)

    # now run the benchmark and measure time
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record(stream)
    graph.launch(stream)
    end.record(stream)
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / niter  # milliseconds


def connect_nvlink(group: mscclpp_comm.CommGroup, remote_nghrs: list):
    for n in remote_nghrs:
        assert type(n) is int
        assert 0 <= n < group.nranks
        assert n != group.my_rank

    tran = Transport.CudaIpc
    connections = group.make_connection(remote_nghrs, tran)
    return connections


class AllgatherKernel:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        connections: dict,
        data: cp.ndarray,
    ):
        self.my_rank = group.my_rank
        self.nranks = group.nranks
        self.data = data

        self.kernel_file = "allgather_kernel.cu"
        self.kernel_name = "allgather_kernel"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()

        channels = group.make_sm_channels(data, connections)
        handles = [channels[rank].device_handle().raw for rank in range(self.nranks) if rank != group.my_rank]
        handles_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

        syncer_arr = cp.zeros((self.nranks - 1) * 12, dtype=cp.bool_)

        assert len(handles) == self.nranks - 1
        assert syncer_arr.shape[0] == (self.nranks - 1) * 12

        self.params = b""
        self.params += struct.pack("P", handles_mem.data.ptr) + struct.pack("P", syncer_arr.data.ptr)

        self._temp = [
            channels, handles, handles_mem, syncer_arr
        ]
        self._data_starts_nelem_totals = {}
        self._params = {}


    def prepare_params(self, n_parallel_sm_blocks, nelem_total):
        if nelem_total in self._data_starts_nelem_totals:
            local_offset, offsets, nelem_per_channel = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.nranks == 0
            nelem_per_channel = nelem_total // self.nranks
            local_offset = self.my_rank * nelem_per_channel
            offsets = cp.array([rank * nelem_per_channel for rank in range(self.nranks) if rank != group.my_rank], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (local_offset, offsets, nelem_per_channel)

        params = self.params + struct.pack("Q", n_parallel_sm_blocks)
        params += struct.pack("Q", local_offset) + struct.pack("P", offsets.data.ptr)
        params += struct.pack("Q", nelem_per_channel)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nblocks, nthreads, nelem_total=None):
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        assert nblocks % (self.nranks - 1) == 0
        n_parallel_sm_blocks = nblocks // (self.nranks - 1)
        params = self.prepare_params(n_parallel_sm_blocks, nelem_total)
        assert nblocks == (self.nranks - 1) * n_parallel_sm_blocks
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, nblocks, nthreads, 0, stream_ptr)


    def __call__(self, nblocks, nthreads, nelem_total=None, stream_ptr=None):
        return self.get_func(nblocks, nthreads, nelem_total)(stream_ptr)


class AllgatherKernelAsync:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        connections: dict,
        data: cp.ndarray,
        proxy_service: ProxyService,
    ):
        self.my_rank = group.my_rank
        self.nranks = group.nranks
        self.data = data

        self.kernel_file = "allgather_kernel_async.cu"
        self.kernel_name = "allgather_kernel"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()

        channels = group.make_proxy_channels(proxy_service, data, connections)
        handles = [channels[rank].device_handle().raw for rank in range(self.nranks) if rank != group.my_rank]
        handles_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

        assert len(handles) == self.nranks - 1

        self.params = b""
        self.params += struct.pack("P", handles_mem.data.ptr) + struct.pack("Q", len(handles))

        self._temp = [
            channels, handles, handles_mem,
        ]
        self._data_starts_nelem_totals = {}
        self._params = {}


    def prepare_params(self, nelem_total):
        if nelem_total in self._data_starts_nelem_totals:
            local_offset, offsets, nelem_per_channel = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.nranks == 0
            nelem_per_channel = nelem_total // self.nranks
            local_offset = self.my_rank * nelem_per_channel
            offsets = cp.array([rank * nelem_per_channel for rank in range(self.nranks) if rank != group.my_rank], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (local_offset, offsets, nelem_per_channel)

        params = self.params + struct.pack("Q", local_offset) + struct.pack("Q", nelem_per_channel)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nblocks, nthreads, nelem_total=None):
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, nblocks, nthreads, 0, stream_ptr)


    def __call__(self, nblocks, nthreads, nelem_total=None, stream_ptr=None):
        return self.get_func(nblocks, nthreads, nelem_total)(stream_ptr)


class AllgatherKernelLL:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        connections: dict,
        data: cp.ndarray,
    ):
        self.my_rank = group.my_rank
        self.nranks = group.nranks
        self.data = data

        self.kernel_file = "allgather_kernel_packet.cu"
        self.kernel_name = "allgather_kernel"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()
        
        scratch = cp.zeros(data.shape[0] * 2, dtype=cp.int32)
        channels = group.make_sm_channels_with_scratch(data, scratch, connections)
        handles = [channels[rank].device_handle().raw for rank in range(self.nranks) if rank != group.my_rank]
        handles_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

        syncer_arr = cp.zeros((self.nranks - 1) * 12, dtype=cp.bool_)

        assert len(handles) == self.nranks - 1
        assert syncer_arr.shape[0] == (self.nranks - 1) * 12

        self.params = b""
        self.params += struct.pack("P", handles_mem.data.ptr) + struct.pack("P", syncer_arr.data.ptr)

        self._temp = [
            channels, handles, handles_mem, syncer_arr, scratch
        ]
        self._data_starts_nelem_totals = {}
        self._params = {}


    def prepare_params(self, n_parallel_sm_blocks, nelem_total):
        if nelem_total in self._data_starts_nelem_totals:
            offsets, local_offset, nelem_per_channel = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.nranks == 0
            nelem_per_channel = nelem_total // self.nranks
            offsets = cp.array([rank * nelem_per_channel for rank in range(self.nranks) if rank != self.my_rank], dtype=cp.uint64)
            local_offset = self.my_rank * nelem_per_channel
            self._data_starts_nelem_totals[nelem_total] = (offsets, local_offset, nelem_per_channel)

        params = self.params + struct.pack("Q", n_parallel_sm_blocks)
        params += struct.pack("P", offsets.data.ptr) + struct.pack("Q", local_offset)
        params += struct.pack("Q", nelem_per_channel) + struct.pack("Q", nelem_total)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nblocks, nthreads, nelem_total=None):
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        assert nblocks % 2 == 0
        assert (nblocks // 2) % (self.nranks - 1) == 0
        n_parallel_sm_blocks = nblocks // 2 // (self.nranks - 1)
        params = self.prepare_params(n_parallel_sm_blocks, nelem_total)
        assert nblocks == (self.nranks - 1) * n_parallel_sm_blocks * 2
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, nblocks, nthreads, 0, stream_ptr)


    def __call__(self, nblocks, nthreads, nelem_total=None, stream_ptr=None):
        return self.get_func(nblocks, nthreads, nelem_total)(stream_ptr)


class ReduceScatterKernel:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        connections: dict,
        data: cp.ndarray,
    ):
        self.my_rank = group.my_rank
        self.nranks = group.nranks
        self.data = data

        self.kernel_file = "reduceScatter_kernel.cu"
        self.kernel_name = "reduceScatter_kernel"
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self._kernel = KernelBuilder(
            file=self.kernel_file,
            kernel_name=self.kernel_name,
            file_dir=file_dir,
        ).get_compiled_kernel()

        channels = group.make_sm_channels(data, connections)
        handles = [channels[rank].device_handle().raw for rank in range(self.nranks) if rank != self.my_rank]
        handles_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

        assert len(handles) == self.nranks - 1

        self.params = b""
        self.params += struct.pack("P", handles_mem.data.ptr)
        self.params += struct.pack("Q", self.my_rank) + struct.pack("Q", self.nranks)
        self.params += struct.pack("P", data.data.ptr)

        self._temp = [
            channels, handles, handles_mem
        ]
        self._data_starts_nelem_totals = {}
        self._params = {}


    def prepare_params(self, n_parallel_sm_blocks, nelem_total):
        if nelem_total in self._data_starts_nelem_totals:
            nelem_per_channel = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.nranks == 0
            nelem_per_channel = nelem_total // self.nranks
            self._data_starts_nelem_totals[nelem_total] = nelem_per_channel

        params = self.params + struct.pack("Q", nelem_per_channel)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nblocks, nthreads, nelem_total=None):
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        n_parallel_sm_blocks = nblocks
        params = self.prepare_params(n_parallel_sm_blocks, nelem_total)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, nblocks, nthreads, 0, stream_ptr)


    def __call__(self, nblocks, nthreads, nelem_total=None, stream_ptr=None):
        return self.get_func(nblocks, nthreads, nelem_total)(stream_ptr)


def print_row(*args):
    print("".join(f"{arg:>20}" for arg in args), flush=True)


def run_expt(group: mscclpp_comm.CommGroup, func,
             init_data: cp.array, data: cp.array,
             length: int, correctness_check,
             check_iters: int, iters: int):
    group.barrier()
    for _ in range(check_iters):
        cp.copyto(data[:length], init_data)
        group.barrier()
        func()
        cp.cuda.runtime.deviceSynchronize()
        assert correctness_check(), data

    cp.cuda.runtime.deviceSynchronize()
    group.barrier()
    res = bench_time(iters, func) / 1e3
    avg_time = res  # seconds
    min_time = res

    size = length * data.dtype.itemsize
    if group.my_rank == 0:
        print_row(size,
                  f"{avg_time * 1e6:.2f}",
                  f"{min_time * 1e6:.2f}",
                  f"{(size / 1e9) / avg_time:.2f}",
                  f"{(size / 1e9) / min_time:.2f}")
    group.barrier()


def run_allgather(group: mscclpp_comm.CommGroup, connections: dict, data_lengths: list,
                  nblocks: int, nthreads: int,
                  check_iters: int = 10, iters: int = 50):
    # proxy_service = ProxyService()

    data = cp.empty(max(data_lengths), dtype=cp.int32)
    kernel = AllgatherKernel(group, connections, data)
    # kernel = AllgatherKernelAsync(group, connections, data, proxy_service)

    if group.my_rank == 0:
        print(f"Allgather")
        print(f"nblocks={nblocks}, nthreads={nthreads}")
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    
    # proxy_service.start_proxy()

    for length in data_lengths:
        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([off // (length // group.nranks) + 1
                                for off in range(length)], dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[:length], expected)
        else:
            init_data, correctness_check = None, None

        func = kernel.get_func(nblocks, nthreads, nelem_total=length)
        run_expt(group=group, func=func, init_data=init_data, data=data,
                 length=length, correctness_check=correctness_check,
                 check_iters=check_iters, iters=iters)
    
    # proxy_service.stop_proxy()


def run_reduceScatter(group: mscclpp_comm.CommGroup, connections: dict, data_lengths: list,
                      nblocks: int, nthreads: int,
                      check_iters: int = 10, iters: int = 50):
    # proxy_service = ProxyService()

    data = cp.empty(max(data_lengths), dtype=cp.float16)
    kernel = ReduceScatterKernel(group, connections, data)

    if group.my_rank == 0:
        print(f"ReduceScatter")
        print(f"nblocks={nblocks}, nthreads={nthreads}")
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    
    # proxy_service.start_proxy()

    for length in data_lengths:
        shard_size = length // group.nranks
        shard_begin = shard_size * group.my_rank
        shard_end = shard_begin + shard_size

        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.float16)
            expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.float16)
            correctness_check = lambda: cp.array_equal(data[shard_begin: shard_end], expected)
        else:
            init_data, correctness_check = None, None

        func = kernel.get_func(nblocks, nthreads, nelem_total=length)
        run_expt(group=group, func=func, init_data=init_data, data=data,
                 length=length, correctness_check=correctness_check,
                 check_iters=check_iters, iters=iters)
    
    # proxy_service.stop_proxy()


if __name__ == "__main__":
    cp.cuda.Device(MPI.COMM_WORLD.rank).use()
    mpi_group = MpiGroup(list(range(8)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])
    
    run_allgather(
        group, connections,
        nblocks=63, nthreads=128,
        data_lengths=[2 ** 20 * 4 // 4, 2 ** 20 * 12 // 4, 2 ** 20 * 16 // 4],
        # data_lengths=[2 ** 28, 2 ** 29, 2 ** 30],
    )

    del group
