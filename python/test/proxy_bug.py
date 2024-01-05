import cupy as cp
from mscclpp import (
    ProxyService,
    Transport,
)
import mscclpp.comm as mscclpp_comm
from .mscclpp_mpi import MpiGroup
from mscclpp.utils import KernelBuilder
import struct
import os


def create_proxy_channels(proxy_service: ProxyService, group: mscclpp_comm.CommGroup, 
                          nchannels: int, memory: cp.ndarray):
    remote = 0 if group.my_rank == 1 else 1
    connections = group.make_connection([remote], Transport.CudaIpc)
    channels = []
    for _ in range(nchannels):
        channels.append(group.make_proxy_channels(proxy_service, memory, connections)[remote])
    return channels


def main(group: mscclpp_comm.CommGroup, nchannels: int):
    proxy_service = ProxyService()
    channels = create_proxy_channels(proxy_service, group, nchannels, cp.zeros(8, dtype=cp.int32))
    handles = [ch.device_handle().raw for ch in channels]
    channel_mem = cp.asarray(memoryview(b"".join(handles)), dtype=cp.uint8)

    params = b"" + struct.pack("P", channel_mem.data.ptr)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(
        file="proxy_bug_kernel.cu",
        kernel_name="proxy_bug",
        file_dir=file_dir,
    ).get_compiled_kernel()

    nblocks = nchannels
    nthreads = 1024

    proxy_service.start_proxy()
    group.barrier()
    print(f"rank {group.my_rank} running kernel", flush=True)
    kernel.launch_kernel(params, nblocks, nthreads, 0, None)
    cp.cuda.runtime.deviceSynchronize()
    print(f"rank {group.my_rank} done", flush=True)
    group.barrier()
    proxy_service.stop_proxy()


if __name__ == "__main__":
    nchannels = 256

    mpi_group = MpiGroup([0, 1])
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    main(group, nchannels)

    del group
