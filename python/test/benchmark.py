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
from mscclpp_mpi import MpiGroup
from mscclpp import ProxyService
from mscclpp.utils import KernelBuilder


def bench_time(niter: int, func):
    # capture cuda graph for nites of the kernel launch
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for i in range(niter):
            func(stream.ptr)
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
        n_parallel_sm_blocks: int,
    ):
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
        handles = [channels[rank].device_haneld().raw for rank in range(self.nranks) if rank != group.my_rank]
        handles_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

        syncer_arr = cp.zeros((self.nranks - 1) * 12, dtype=cp.bool_)

        assert len(handles) == self.nranks - 1
        assert syncer_arr.shape[0] == (self.nranks - 1) * 12

        self.params = b""
        self.params += struct.pack("P", handles_mem.data.ptr) + struct.pack("P", syncer_arr.data.ptr)
        self.params += struct.pack("i", n_parallel_sm_blocks)

        self._temp = [
            channels, handles, handles_mem, syncer_arr
        ]
        self._data_starts_nelem_totals = {}
        self._params = {}


    def prepare_params(self, nelem_total):
        if nelem_total in self._data_starts_nelem_totals:
            offsets, nelem_per_channel = self._data_starts_nelem_totals[nelem_total]
        else:
            assert nelem_total % self.nranks == 0
            nelem_per_channel = nelem_total // self.nranks
            offsets = cp.array([rank * nelem_per_channel for rank in range(self.nranks) if rank != group.my_rank], dtype=cp.uint64)
            self._data_starts_nelem_totals[nelem_total] = (offsets, nelem_per_channel)

        params = self.params + struct.pack("P", offsets.data.ptr) + struct.pack("Q", nelem_per_channel)
        self._params[uuid.uuid1()] = params

        return params


    def get_func(self, nblocks, nthreads, nelem_total=None):
        if nelem_total is None:
            nelem_total = self.data.shape[0]
        params = self.prepare_params(nelem_total)
        return lambda stream_ptr=None, params=params: self._kernel.launch_kernel(params, nblocks, nthreads, 0, stream_ptr)


    def __call__(self, nblocks, nthreads, nelem_total=None, stream_ptr=None):
        return self.get_func(nblocks, nthreads, nelem_total)(stream_ptr)


def run_allgather(group: mscclpp_comm.CommGroup, connections: dict, data_size: int):
    pass


if __name__ == "__main__":
    cp.cuda.Device(MPI.COMM_WORLD.rank).use()
    mpi_group = MpiGroup(list(range(8)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])
    


