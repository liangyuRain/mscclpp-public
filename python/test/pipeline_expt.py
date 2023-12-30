import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np

import mscclpp.comm as mscclpp_comm
from .pipeline_schedule import allreduce_kernel, allgather_kernel, connect_nvlink, KERNEL_FILE
from .mscclpp_mpi import MpiGroup


def print_row(*args):
    print("".join(f"{arg:>20}" for arg in args), flush=True)


def run_allreduce(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup, connections: dict,
                  data_lengths: list, nelem_per_send: int, scratch_size: int,
                  check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 25 + " Allreduce " + "#" * 25)
        print(f"k={k}, nelem_per_send={nelem_per_send}, scratch_size={scratch_size},\n"
              f"check_iters={check_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}")
        print()
        print_row("size (B)", "time (us)", "algbw (GB/s)")
    for length in data_lengths:
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)

        init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
        expected = cp.array([sum(n + 1 for n in range(group.nranks))] * length, dtype=cp.int32)

        data = cp.zeros(length, dtype=cp.int32)

        kernel = allreduce_kernel(Ts, Cs, k,
                                  group=group,
                                  connections=connections,
                                  data=data,
                                  allreduce_length=length,
                                  nelem_per_send=nelem_per_send,
                                  scratch_size=scratch_size)

        for _ in range(check_iters):
            group.barrier()  # Add barrier here to prevent initialization overwrites the data
                             # another gpu is still getting in previous iter.
            cp.copyto(data, init_data)
            kernel()
            assert cp.array_equal(data, expected)
            cp.cuda.runtime.deviceSynchronize()

        group.barrier()
        res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
        time_span = np.average(res)  # seconds

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{time_span * 1e6:.2f}", 
                      f"{(size / 2 ** 30) / time_span:.2f}")


def run_allgather(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup, connections: dict,
                  data_lengths: list, nelem_per_send: int, check_iters: int = 10,
                  warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 25 + " Allgather " + "#" * 25)
        print(f"k={k}, nelem_per_send={nelem_per_send},\n"
              f"check_iters={check_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}")
        print()
        print_row("size(B)", "time(us)", "algbw(GB/s)")
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
                                  data=data,
                                  allgather_length=length,
                                  nelem_per_send=nelem_per_send)

        for _ in range(check_iters):
            group.barrier()
            cp.copyto(data, init_data)
            kernel()
            assert cp.array_equal(data, expected)
            cp.cuda.runtime.deviceSynchronize()

        group.barrier()
        res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
        time_span = np.average(res)  # seconds

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{time_span * 1e6:.2f}", 
                      f"{(size / 2 ** 30) / time_span:.2f}")


if __name__ == "__main__":
    mpi_group = MpiGroup(list(range(8)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    # allpairs
    k = 8
    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])

    run_allgather(Ts, Cs, k, group=group, connections=connections, 
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 20,
                  warmup_iters=20,
                  iters=50)

    if group.my_rank == 0:
        print()

    # ring
    k = 16
    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [(group.my_rank - 1) % group.nranks,
                                         (group.my_rank + 1) % group.nranks])

    run_allreduce(Ts, Cs, k, group=group, connections=connections, 
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 15,
                  scratch_size=2 ** 22,
                  warmup_iters=20,
                  iters=50)

    del group
