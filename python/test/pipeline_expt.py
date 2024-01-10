import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np

import mscclpp.comm as mscclpp_comm
from .pipeline_schedule import (
    allreduce_kernel,
    allgather_kernel,
    reduce_scatter_kernel,
    connect_nvlink,
    KERNEL_FILE,
)
from .mscclpp_mpi import MpiGroup
from mscclpp import ProxyService


def print_row(*args):
    print("".join(f"{arg:>20}" for arg in args), flush=True)


def run_allreduce(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, nelem_per_send: int, scratch_size: int,
                  check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    proxy_service = ProxyService()
    if group.my_rank == 0:
        print("#" * 45 + " Allreduce " + "#" * 45)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={check_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)

        init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
        expected = cp.array([sum(n + 1 for n in range(group.nranks))] * length, dtype=cp.int32)

        data = cp.zeros(length, dtype=cp.int32)

        kernel = allreduce_kernel(Ts, Cs, k,
                                  group=group,
                                  connections=connections,
                                  connection_types=connection_types,
                                  data=data,
                                  allreduce_length=length,
                                  nelem_per_send=nelem_per_send,
                                  scratch_size=scratch_size,
                                  proxy_service=proxy_service)

        proxy_service.start_proxy()

        for _ in range(check_iters):
            cp.copyto(data, init_data)
            kernel()
            cp.cuda.runtime.deviceSynchronize()
            assert cp.array_equal(data, expected)

        group.barrier()
        res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
        avg_time = np.average(res)  # seconds
        min_time = np.min(res)

        proxy_service.stop_proxy()

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{avg_time * 1e6:.2f}",
                      f"{min_time * 1e6:.2f}",
                      f"{(size / 1e9) / avg_time:.2f}",
                      f"{(size / 1e9) / min_time:.2f}")


def run_allgather(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, nelem_per_send: int, check_iters: int = 10,
                  warmup_iters: int = 10, iters: int = 10):
    proxy_service = ProxyService()
    if group.my_rank == 0:
        print("#" * 45 + " Allgather " + "#" * 45)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}")
        print(f"check_iters={check_iters}, warmup_iters={check_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)

        init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
        expected = cp.array([off // (length // group.nranks) + 1
                             for off in range(length)], dtype=cp.int32)

        data = cp.zeros(length, dtype=cp.int32)

        kernel = allgather_kernel(Ts, Cs, k,
                                  group=group,
                                  connections=connections,
                                  connection_types=connection_types,
                                  data=data,
                                  allgather_length=length,
                                  nelem_per_send=nelem_per_send,
                                  proxy_service=proxy_service)

        proxy_service.start_proxy()

        for _ in range(check_iters):
            cp.copyto(data, init_data)
            kernel()
            cp.cuda.runtime.deviceSynchronize()
            assert cp.array_equal(data, expected)

        group.barrier()
        res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
        avg_time = np.average(res)  # seconds
        min_time = np.min(res)

        proxy_service.stop_proxy()

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{avg_time * 1e6:.2f}",
                      f"{min_time * 1e6:.2f}",
                      f"{(size / 1e9) / avg_time:.2f}",
                      f"{(size / 1e9) / min_time:.2f}")


def run_reduce_scatter(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                       connections: dict, connection_types: dict,
                       data_lengths: list, nelem_per_send: int, scratch_size: int,
                       check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    proxy_service = ProxyService()
    if group.my_rank == 0:
        print("#" * 43 + " ReduceScatter " + "#" * 43)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={check_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)
        
        assert length % group.nranks == 0
        shard_size = length // group.nranks
        shard_begin = shard_size * group.my_rank
        shard_end = shard_begin + shard_size

        init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
        expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)

        data = cp.zeros(length, dtype=cp.int32)

        kernel = reduce_scatter_kernel(Ts, Cs, k,
                                       group=group,
                                       connections=connections,
                                       connection_types=connection_types,
                                       data=data,
                                       reduce_scatter_length=length,
                                       nelem_per_send=nelem_per_send,
                                       scratch_size=scratch_size,
                                       proxy_service=proxy_service)

        proxy_service.start_proxy()

        for _ in range(check_iters):
            cp.copyto(data, init_data)
            kernel()
            cp.cuda.runtime.deviceSynchronize()
            assert cp.array_equal(data[shard_begin: shard_end], expected)

        group.barrier()
        res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
        avg_time = np.average(res)  # seconds
        min_time = np.min(res)

        proxy_service.stop_proxy()

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{avg_time * 1e6:.2f}",
                      f"{min_time * 1e6:.2f}",
                      f"{(size / 1e9) / avg_time:.2f}",
                      f"{(size / 1e9) / min_time:.2f}")


if __name__ == "__main__":
    mpi_group = MpiGroup(list(range(8)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    def channel_type(dest):
        tp = "sm"
        # tp = "proxy"
        # tp = "sm" if dest // 4 == group.my_rank // 4 else "proxy"
        return tp

    # allpairs
    k = 4
    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])

    run_allgather(Ts, Cs, k, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 18,
                  warmup_iters=20,
                  iters=50)

    if group.my_rank == 0:
        print()

    # ring
    k = 8
    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [(group.my_rank - 1) % group.nranks,
                                         (group.my_rank + 1) % group.nranks])

    run_reduce_scatter(Ts, Cs, k, group=group, connections=connections, 
                       connection_types={dest: channel_type(dest) for dest in connections},
                       # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                       data_lengths=[2 ** 28],
                       nelem_per_send=2 ** 18,
                       scratch_size=2 ** 20,
                       warmup_iters=20,
                       iters=50)

    if group.my_rank == 0:
        print()

    # ring
    k = 4
    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [(group.my_rank - 1) % group.nranks,
                                         (group.my_rank + 1) % group.nranks])

    run_allreduce(Ts, Cs, k, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 15,
                  scratch_size=2 ** 20,
                  warmup_iters=20,
                  iters=50)

    del group
