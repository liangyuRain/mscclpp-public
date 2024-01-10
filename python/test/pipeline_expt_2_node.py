import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np
import pickle
import copy
from mpi4py import MPI
import os

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
from mscclpp import ProxyService, Transport


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


def run_allreduce(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, nelem_per_send: int, scratch_size: int,
                  check_iters: int = 10, warmup_iters: int = 10, iters: int = 10):
    if group.my_rank == 0:
        print("#" * 45 + " Allreduce " + "#" * 45)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        proxy_service = ProxyService()
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
        if BENCH_METHOD == 1:
            res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
            avg_time = np.average(res)  # seconds
            min_time = np.min(res)
        elif BENCH_METHOD == 2:
            res = bench_time(iters, kernel) / 1e3
            avg_time = res  # seconds
            min_time = res
        else:
            raise ValueError(f"Unknown BENCH_METHOD: {BENCH_METHOD}")

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
    if group.my_rank == 0:
        print("#" * 45 + " Allgather " + "#" * 45)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        proxy_service = ProxyService()
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
        if BENCH_METHOD == 1:
            res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
            avg_time = np.average(res)  # seconds
            min_time = np.min(res)
        elif BENCH_METHOD == 2:
            res = bench_time(iters, kernel) / 1e3
            avg_time = res  # seconds
            min_time = res
        else:
            raise ValueError(f"Unknown BENCH_METHOD: {BENCH_METHOD}")

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
    if group.my_rank == 0:
        print("#" * 43 + " ReduceScatter " + "#" * 43)
        print(f"nranks={group.nranks}")
        print(f"k={k}, nelem_per_send={nelem_per_send}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"KERNEL_FILE={KERNEL_FILE}, BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")
    for length in data_lengths:
        proxy_service = ProxyService()
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
        if BENCH_METHOD == 1:
            res = benchmark(lambda: kernel(), n_warmup=warmup_iters, n_repeat=iters).gpu_times
            avg_time = np.average(res)  # seconds
            min_time = np.min(res)
        elif BENCH_METHOD == 2:
            res = bench_time(iters, kernel) / 1e3
            avg_time = res  # seconds
            min_time = res
        else:
            raise ValueError(f"Unknown BENCH_METHOD: {BENCH_METHOD}")

        proxy_service.stop_proxy()

        size = length * 4
        if group.my_rank == 0:
            print_row(size, 
                      f"{avg_time * 1e6:.2f}",
                      f"{min_time * 1e6:.2f}",
                      f"{(size / 1e9) / avg_time:.2f}",
                      f"{(size / 1e9) / min_time:.2f}")


def multi_instance(Ts: dict, Cs: dict, k: int, ninstance: int):
    rTs, rCs = {}, {}
    count = {}
    for (u, i), l in Ts.items():
        C = Cs[u, i]
        for j in range(C * ninstance):
            ri = count.get(u, 0)
            count[u] = ri + 1
            rTs[u, ri] = copy.deepcopy(l)
            rCs[u, ri] = 1
    assert all(count[u] == k * ninstance for u, _ in Ts.keys())
    return rTs, rCs, k * ninstance


def test_peer_to_peer(group: mscclpp_comm.CommGroup, connections: dict, src: int, dest: int):
    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    nelem_total = 2 ** 28
    nelem_per_send = 2 ** 18
    nblocks = 24
    assert nelem_total % nblocks == 0
    shard_size = nelem_total // nblocks
    proxy_service = ProxyService()
    if group.my_rank == src:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        if group.my_rank // 8 == dest // 8:
            send_sm_channels = {bid: [group.make_sm_channel(memory, connections[dest], dest)]
                                for bid in range(nblocks)}
        else:
            send_proxy_channels = {bid: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]
                                   for bid in range(nblocks)}
    elif group.my_rank == dest:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        if group.my_rank // 8 == src // 8:
            recv_sm_channels = {bid: [group.make_sm_channel(memory, connections[src], src)]
                                for bid in range(nblocks)}
        else:
            recv_proxy_channels = {bid: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]
                                   for bid in range(nblocks)}
    else:
        pass

    data_offsets = {bid: shard_size * bid for bid in range(nblocks)}
    data_sizes = {bid: shard_size for bid in range(nblocks)}
    scratch_size = 0
    node_types = {bid: 1 for bid in range(nblocks)}
    
    if group.my_rank == src or group.my_rank == dest:
        kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                memory, data_offsets, data_sizes, scratch_size, {}, {}, node_types, 
                                nelem_per_send, nblocks)
    else:
        kernel = None

    proxy_service.start_proxy()
    group.barrier()
    if kernel is not None:
        res = benchmark(lambda: kernel(), n_warmup=100, n_repeat=100).gpu_times
    else:
        res = [-1]
    avg_time = np.average(res)  # seconds
    min_time = np.min(res)

    group.barrier()
    proxy_service.stop_proxy()

    size = nelem_total * 4
    if group.my_rank == src:
        print_row(str((src, dest)),
                size, 
                f"{avg_time * 1e6:.2f}",
                f"{min_time * 1e6:.2f}",
                f"{(size / 2 ** 30) / avg_time:.2f}",
                f"{(size / 2 ** 30) / min_time:.2f}")
    if kernel is not None:
        assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))


if __name__ == "__main__":
    if MPI.COMM_WORLD.rank < 8:
        os.environ["MSCCLPP_HCA_DEVICES"] = ",".join([f"mlx5_{i}" for i in range(9) if i != 3])
    else:
        os.environ["MSCCLPP_HCA_DEVICES"] = ",".join([f"mlx5_{i}" for i in range(9) if i != 2])
    cp.cuda.Device(MPI.COMM_WORLD.rank % 8).use()

    mpi_group = MpiGroup(list(range(16)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    def channel_type(dest):
        tp = "sm" if dest // 8 == group.my_rank // 8 else "proxy"
        return tp

    remote_nghrs = [v for v in range(group.nranks) if v != group.my_rank]
    connections = group.make_connection(remote_nghrs,
                                        {v: Transport.CudaIpc if v // 8 == group.my_rank // 8
                                            else group.my_ib_device(group.my_rank % 8)
                                         for v in remote_nghrs})

    k = 4
    # tree_name = f"adjusted_bw_k_{k}_293"
    tree_name = f"adjusted_noPCIe_IB20_NV250_bw_k_{k}_286"
    if group.my_rank == 0:
        print(f"tree_file={tree_name}")
    with open(f"/root/mscclpp-public/trees/{tree_name}.pkl", "rb") as f:
        Ts, Cs = pickle.load(f)

    # Allgather
    ninstance = 1
    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
    run_allgather(Tsp, Csp, kp, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 18,
                  warmup_iters=20,
                  iters=50)

    if group.my_rank == 0:
        print()

    # ReduceScatter
    ninstance = 1
    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
    run_reduce_scatter(Tsp, Csp, kp, group=group, connections=connections, 
                       connection_types={dest: channel_type(dest) for dest in connections},
                       # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                       data_lengths=[2 ** 28],
                       nelem_per_send=2 ** 18,
                       scratch_size=2 ** 20,
                       warmup_iters=20,
                       iters=50)

    if group.my_rank == 0:
        print()

    # Allreduce
    ninstance = 1
    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
    run_allreduce(Tsp, Csp, kp, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  # data_lengths=[2 ** (n - 2) for n in range(10, 31)],
                  data_lengths=[2 ** 28],
                  nelem_per_send=2 ** 15,
                  scratch_size=2 ** 20,
                  warmup_iters=20,
                  iters=50)

    del group
