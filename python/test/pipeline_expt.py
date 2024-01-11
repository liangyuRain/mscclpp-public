import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np
from mpi4py import MPI

import mscclpp.comm as mscclpp_comm
from pipeline_schedule import (
    PipelineKernel,
    allreduce_kernel,
    allgather_kernel,
    reduce_scatter_kernel,
    connect_nvlink,
    KERNEL_FILE,
)
from mscclpp_mpi import MpiGroup
from mscclpp import ProxyService


BENCH_METHOD = 2


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


def print_row(*args):
    print("".join(f"{arg:>20}" for arg in args), flush=True)


def run_expt(group: mscclpp_comm.CommGroup, kernel: PipelineKernel,
             init_data: cp.array, data: cp.array,
             length: int, nelem_per_send: int, k: int,
             correctness_check,
             check_iters: int, warmup_iters: int, iters: int):
    func = kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
    group.barrier()
    for _ in range(check_iters):
        cp.copyto(data[:length], init_data)
        func()
        cp.cuda.runtime.deviceSynchronize()
        assert correctness_check()

    cp.cuda.runtime.deviceSynchronize()
    group.barrier()
    if BENCH_METHOD == 1:
        res = benchmark(func, n_warmup=warmup_iters, n_repeat=iters).gpu_times
        avg_time = np.average(res)  # seconds
        min_time = np.min(res)
    elif BENCH_METHOD == 2:
        res = bench_time(iters, func) / 1e3
        avg_time = res  # seconds
        min_time = res
    else:
        raise ValueError(f"Unknown BENCH_METHOD: {BENCH_METHOD}")

    size = length * 4
    send_size = nelem_per_send * 4
    if group.my_rank == 0:
        print_row(size, send_size,
                  f"{avg_time * 1e6:.2f}",
                  f"{min_time * 1e6:.2f}",
                  f"{(size / 1e9) / avg_time:.2f}",
                  f"{(size / 1e9) / min_time:.2f}")
    group.barrier()


def run_allreduce(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, send_lengths: list, scratch_size: int,
                  check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 55 + " Allreduce " + "#" * 55)
        print(f"nranks={group.nranks}")
        print(f"k={k}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service = ProxyService()

    max_length = max(data_lengths)
    data = cp.zeros(max_length, dtype=cp.int32)
    kernel = allreduce_kernel(Ts, Cs, k,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=data,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)

        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([sum(n + 1 for n in range(group.nranks))] * length, dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[:length], expected)
        else:
            init_data, correctness_check = None, None

        run_expt(group=group, kernel=kernel, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send, k=k,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters)

    proxy_service.stop_proxy()


def run_allgather(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, send_lengths: list, check_iters: int = 10,
                  warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 55 + " Allgather " + "#" * 55)
        print(f"nranks={group.nranks}")
        print(f"k={k}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service = ProxyService()

    max_length = max(data_lengths)
    data = cp.zeros(max_length, dtype=cp.int32)
    kernel = allgather_kernel(Ts, Cs, k,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=data,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)

        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([off // (length // group.nranks) + 1
                                for off in range(length)], dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[:length], expected)
        else:
            init_data, correctness_check = None, None

        run_expt(group=group, kernel=kernel, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send, k=k,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters)

    proxy_service.stop_proxy()


def run_reduce_scatter(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                       connections: dict, connection_types: dict,
                       data_lengths: list, send_lengths: list, scratch_size: int,
                       check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 53 + " ReduceScatter " + "#" * 53)
        print(f"nranks={group.nranks}")
        print(f"k={k}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service = ProxyService()

    max_length = max(data_lengths)
    data = cp.zeros(max_length, dtype=cp.int32)
    kernel = reduce_scatter_kernel(Ts, Cs, k,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=data,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % (k * group.nranks) != 0:
            length = math.ceil(length / (k * group.nranks)) * (k * group.nranks)
        
        assert length % group.nranks == 0
        shard_size = length // group.nranks
        shard_begin = shard_size * group.my_rank
        shard_end = shard_begin + shard_size

        if check_iters > 0:    
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[shard_begin: shard_end], expected)
        else:
            init_data, correctness_check = None, None

        run_expt(group=group, kernel=kernel, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send, k=k,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters)

    proxy_service.stop_proxy()


if __name__ == "__main__":
    cp.cuda.Device(MPI.COMM_WORLD.rank).use()
    mpi_group = MpiGroup(list(range(8)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    def channel_type(dest):
        tp = "sm"
        # tp = "proxy"
        # tp = "sm" if dest // 4 == group.my_rank // 4 else "proxy"
        return tp

    data_lengths=[2 ** (n - 2) for n in range(20, 31)]
    check_iters = 10
    warmup_iters = 20
    bench_iters = 50

    # allpairs
    k = 4
    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(k))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(k))}
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])

    run_allgather(Ts, Cs, k, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  data_lengths=data_lengths,
                  send_lengths=[2 ** 18],
                  check_iters=check_iters,
                  warmup_iters=warmup_iters,
                  iters=bench_iters)

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
                       data_lengths=data_lengths,
                       send_lengths=[2 ** 18],
                       scratch_size=2 ** 20,
                       check_iters=check_iters,
                       warmup_iters=warmup_iters,
                       iters=bench_iters)

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
                  data_lengths=data_lengths,
                  send_lengths=[2 ** 15],
                  scratch_size=2 ** 20,
                  check_iters=check_iters,
                  warmup_iters=warmup_iters,
                  iters=bench_iters)

    del group
