import cupy as cp
from cupyx.profiler import benchmark
import itertools
import math
import numpy as np
from mpi4py import MPI
import pickle

import mscclpp.comm as mscclpp_comm
from .pipeline_schedule import (
    PipelineKernel,
    allreduce_kernel,
    allgather_kernel,
    reduce_scatter_kernel,
    reduce_scatter_kernel_hack,
    connect_nvlink,
)
from .mscclpp_mpi import MpiGroup
from mscclpp import ProxyService


BENCH_METHOD = 1

DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_9 = {
    0: 1,
    1: 11,
    2: 0,
    3: 10,
    4: 4,
    5: 3,
    6: 5,
    7: 14,
    8: 9,
    9: 8,
    10: 13,
    11: 7,
    12: 12,
    13: 6,
    14: 2,
    15: 15,
}

DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_8 = {
    0: 9,
    1: 8,
    2: 1,
    3: 0,
    4: 2,
    5: 13,
    6: 10,
    7: 15,
    8: 6,
    9: 3,
    10: 4,
    11: 5,
    12: 14,
    13: 7,
    14: 11,
    15: 12,
}


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


def run_expt(group: mscclpp_comm.CommGroup, func,
             init_data: cp.array, data: cp.array,
             length: int, nelem_per_send: int,
             correctness_check,
             check_iters: int, warmup_iters: int, iters: int,
             skip_leaf_tb: bool = False):
    group.barrier()
    for _ in range(check_iters):
        cp.copyto(data[:length], init_data)
        if skip_leaf_tb:
            cp.cuda.runtime.deviceSynchronize()
            group.barrier()
        func()
        cp.cuda.runtime.deviceSynchronize()
        if skip_leaf_tb:
            group.barrier()
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
    if type(nelem_per_send) is tuple:
        send_size = ','.join(str(size * 4) for size in nelem_per_send)
    elif type(nelem_per_send) is int:
        send_size = nelem_per_send * 4
    else:
        assert False, type(nelem_per_send)
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
    proxy_service = ProxyService()

    alignment = 4 * k * group.nranks
    max_length = max(math.ceil(length / alignment) * alignment for length in data_lengths)
    data = cp.empty(max_length, dtype=cp.int32)
    kernel = allreduce_kernel(Ts, Cs, k,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=data,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    if group.my_rank == 0:
        print("#" * 55 + " Allreduce " + "#" * 55)
        print(f"nranks={group.nranks}")
        print(f"k={k}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"nblocks={kernel.nblocks}")
        print(f"KERNEL={kernel.kernel_file}::{kernel.kernel_name}")
        print(f"BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % alignment != 0:
            length = math.ceil(length / alignment) * alignment

        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([sum(n + 1 for n in range(group.nranks))] * length, dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[:length], expected)
        else:
            init_data, correctness_check = None, None

        func = kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
        run_expt(group=group, func=func, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters)

    proxy_service.stop_proxy()


def run_fusion_allreduce(reduce_scatter_Ts: dict, reduce_scatter_Cs: dict, reduce_scatter_k: int,
                         allgather_Ts: dict, allgather_Cs: dict, allgather_k: int,
                         group: mscclpp_comm.CommGroup,
                         connections: dict, connection_types: dict,
                         data_lengths: list, rs_send_lengths: dict, ag_send_lengths: dict, 
                         scratch_size: int, check_iters: int = 10, warmup_iters: int = 10, iters: int = 10,
                         use_reduceScatter_kernel: bool = False, rs_n_parallel_sm_blocks: int = 1,
                         rs_n_parallel_reduce_blocks: int = None, skip_leaf_tb: bool = True,
                         coll_re: bool = False, sendtb: bool = False,
                         ag_n_parallel_sm_blocks: int = 1):
    proxy_service = ProxyService()

    lcm = reduce_scatter_k * allgather_k // math.gcd(reduce_scatter_k, allgather_k)
    alignment = 4 * lcm * group.nranks
    max_length = max(math.ceil(length / alignment) * alignment for length in data_lengths)
    data = cp.empty(max_length, dtype=cp.int32)
    RS_kernel = reduce_scatter_kernel(reduce_scatter_Ts, reduce_scatter_Cs, reduce_scatter_k,
                                      group=group,
                                      connections=connections,
                                      connection_types=connection_types,
                                      data=data,
                                      scratch_size=scratch_size,
                                      proxy_service=proxy_service,
                                      use_reduceScatter_kernel=use_reduceScatter_kernel,
                                      n_parallel_sm_blocks=rs_n_parallel_sm_blocks,
                                      n_parallel_reduce_blocks=rs_n_parallel_reduce_blocks,
                                      skip_leaf_tb=skip_leaf_tb,
                                      coll_re=coll_re,
                                      sendtb=sendtb)
    AG_kernel = allgather_kernel(allgather_Ts, allgather_Cs, allgather_k,
                                 group=group,
                                 connections=connections,
                                 connection_types=connection_types,
                                 data=data,
                                 proxy_service=proxy_service,
                                 n_parallel_sm_blocks=ag_n_parallel_sm_blocks)

    if group.my_rank == 0:
        print("#" * 55 + " Allreduce " + "#" * 55)
        print(f"nranks={group.nranks}")
        print(f"RS_k={reduce_scatter_k}, AG_k={allgather_k}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"use_reduceScatter_kernel={use_reduceScatter_kernel}")
        print(f"skip_leaf_tb={skip_leaf_tb}")
        print(f"rs_nblocks={RS_kernel.nblocks}, rs_n_parallel_sm_blocks={rs_n_parallel_sm_blocks}, rs_n_parallel_reduce_blocks={rs_n_parallel_reduce_blocks}")
        print(f"ag_nblocks={AG_kernel.nblocks}, ag_n_parallel_sm_blocks={ag_n_parallel_sm_blocks}")
        print(f"RS_KERNEL={RS_kernel.kernel_file}::{RS_kernel.kernel_name}")
        print(f"AG_KERNEL={AG_kernel.kernel_file}::{AG_kernel.kernel_name}")
        print(f"BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service.start_proxy()

    for length in data_lengths:
        for rs_nelem_per_send, ag_nelem_per_send in itertools.product(rs_send_lengths[length], ag_send_lengths[length]):
            if length % alignment != 0:
                length = math.ceil(length / alignment) * alignment

            if check_iters > 0:
                init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
                expected = cp.array([sum(n + 1 for n in range(group.nranks))] * length, dtype=cp.int32)
                correctness_check = lambda: cp.array_equal(data[:length], expected)
            else:
                init_data, correctness_check = None, None

            RS_func = RS_kernel.get_func(nelem_total=length, nelem_per_send=rs_nelem_per_send)
            AG_func = AG_kernel.get_func(nelem_total=length, nelem_per_send=ag_nelem_per_send)

            def func(stream_ptr=None):
                RS_func(stream_ptr)
                AG_func(stream_ptr)

            run_expt(group=group, func=func, init_data=init_data, data=data,
                    length=length, nelem_per_send=(rs_nelem_per_send, ag_nelem_per_send),
                    correctness_check=correctness_check,
                    check_iters=check_iters, warmup_iters=warmup_iters, iters=iters,
                    skip_leaf_tb=skip_leaf_tb)

    proxy_service.stop_proxy()


def run_allgather(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                  connections: dict, connection_types: dict,
                  data_lengths: list, send_lengths: list, check_iters: int = 10,
                  warmup_iters: int = 10, iters: int = 10,
                  n_parallel_sm_blocks: int = 2):
    proxy_service = ProxyService()

    alignment = 4 * k * group.nranks
    max_length = max(math.ceil(length / alignment) * alignment for length in data_lengths)
    data = cp.empty(max_length, dtype=cp.int32)
    kernel = allgather_kernel(Ts, Cs, k,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=data,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    if group.my_rank == 0:
        print("#" * 55 + " Allgather " + "#" * 55)
        print(f"nranks={group.nranks}")
        print(f"k={k}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"nblocks={kernel.nblocks}, n_parallel_sm_blocks={n_parallel_sm_blocks}")
        print(f"KERNEL={kernel.kernel_file}::{kernel.kernel_name}")
        print(f"BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % alignment != 0:
            length = math.ceil(length / alignment) * alignment

        if check_iters > 0:
            init_data = cp.array([group.my_rank + 1] * length, dtype=cp.int32)
            expected = cp.array([off // (length // group.nranks) + 1
                                for off in range(length)], dtype=cp.int32)
            correctness_check = lambda: cp.array_equal(data[:length], expected)
        else:
            init_data, correctness_check = None, None

        func = kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
        run_expt(group=group, func=func, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters)

    proxy_service.stop_proxy()


def run_reduce_scatter(Ts: dict, Cs: dict, k: int, group: mscclpp_comm.CommGroup,
                       connections: dict, connection_types: dict,
                       data_lengths: list, send_lengths: list, scratch_size: int,
                       check_iters: int = 10, warmup_iters: int = 10, iters: int = 10,
                       use_reduceScatter_kernel=False, n_parallel_sm_blocks: int = 1,
                       n_parallel_reduce_blocks: int = None, coll_re: bool = False,
                       skip_leaf_tb: bool = False, hack: bool = False, sendtb: bool= False,
                       n_pipeline: int = None):
    proxy_service = ProxyService()

    alignment = 4 * k * group.nranks
    max_length = max(math.ceil(length / alignment) * alignment for length in data_lengths)
    data = cp.empty(max_length, dtype=cp.int32)
    assert not (hack and sendtb)
    if hack:
        kernel = reduce_scatter_kernel_hack(Ts, Cs, k,
                                            group=group,
                                            connections=connections,
                                            connection_types=connection_types,
                                            data=data,
                                            scratch_size=scratch_size,
                                            proxy_service=proxy_service,
                                            n_parallel_sm_blocks=n_parallel_sm_blocks,
                                            skip_leaf_tb=skip_leaf_tb)
    else:
        kernel = reduce_scatter_kernel(Ts, Cs, k,
                                       group=group,
                                       connections=connections,
                                       connection_types=connection_types,
                                       data=data,
                                       scratch_size=scratch_size,
                                       proxy_service=proxy_service,
                                       use_reduceScatter_kernel=use_reduceScatter_kernel,
                                       n_parallel_sm_blocks=n_parallel_sm_blocks,
                                       n_parallel_reduce_blocks=n_parallel_reduce_blocks,
                                       skip_leaf_tb=skip_leaf_tb,
                                       coll_re=coll_re,
                                       sendtb=sendtb,
                                       n_pipeline=n_pipeline)
    
    if group.my_rank == 0:
        print("#" * 53 + " ReduceScatter " + "#" * 53)
        print(f"nranks={group.nranks}")
        print(f"k={k}, scratch_size={scratch_size}")
        print(f"check_iters={check_iters}, warmup_iters={warmup_iters}, iters={iters}")
        print(f"use_reduceScatter_kernel={use_reduceScatter_kernel}")
        print(f"skip_leaf_tb={skip_leaf_tb}, n_pipeline={n_pipeline}")
        print(f"nblocks={kernel.nblocks}, n_parallel_sm_blocks={n_parallel_sm_blocks}, n_parallel_reduce_blocks={n_parallel_reduce_blocks}")
        print(f"KERNEL={kernel.kernel_file}::{kernel.kernel_name}")
        print(f"BENCH_METHOD={BENCH_METHOD}")
        print()
        print_row("size(B)", "send_size(B)", "avg_time(us)", "min_time(us)", "avg_algbw(GB/s)", "max_algbw(GB/s)")

    proxy_service.start_proxy()

    for length, nelem_per_send in itertools.product(data_lengths, send_lengths):
        if length % alignment != 0:
            length = math.ceil(length / alignment) * alignment
        
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

        func = kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
        run_expt(group=group, func=func, init_data=init_data, data=data,
                 length=length, nelem_per_send=nelem_per_send,
                 correctness_check=correctness_check,
                 check_iters=check_iters, warmup_iters=warmup_iters, iters=iters,
                 skip_leaf_tb=skip_leaf_tb)

    proxy_service.stop_proxy()


def test_peer_to_peer(group: mscclpp_comm.CommGroup, connections: dict, src: int, dest: int):
    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    nelem_total = 3 * 2 ** 25
    nelem_per_send = 2 ** 18
    nblocks = 24
    assert nelem_total % nblocks == 0
    shard_size = nelem_total // nblocks
    proxy_service = ProxyService()
    if group.my_rank == src:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        if group.my_rank // 16 == dest // 16:
            send_sm_channels = {bid: [group.make_sm_channel(memory, connections[dest], dest)]
                                for bid in range(nblocks)}
        else:
            send_proxy_channels = {bid: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]
                                   for bid in range(nblocks)}
    elif group.my_rank == dest:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        if group.my_rank // 16 == src // 16:
            recv_sm_channels = {bid: [group.make_sm_channel(memory, connections[src], src)]
                                for bid in range(nblocks)}
        else:
            recv_proxy_channels = {bid: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]
                                   for bid in range(nblocks)}
    else:
        pass

    data_chunk_offsets = {bid: shard_size * bid for bid in range(nblocks)}
    data_chunk_sizes = {bid: shard_size for bid in range(nblocks)}
    total_chunks = nblocks
    scratch_size = 0
    node_types = {bid: 1 for bid in range(nblocks)}
    
    if group.my_rank == src or group.my_rank == dest:
        kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size, {}, {},
                                node_types, nblocks)
    else:
        kernel = None

    proxy_service.start_proxy()
    group.barrier()
    if kernel is not None:
        res = benchmark(kernel.get_func(nelem_total, nelem_per_send), n_warmup=100, n_repeat=100).gpu_times
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
    # if kernel is not None:
    #     assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))


if __name__ == "__main__":
    cp.cuda.Device(MPI.COMM_WORLD.rank).use()
    mpi_group = MpiGroup(list(range(16)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)
    connections = connect_nvlink(group, [v for v in range(group.nranks) 
                                         if v != group.my_rank])

    def channel_type(dest):
        tp = "sm"
        # tp = "proxy"
        # tp = "sm" if dest // 4 == group.my_rank // 4 else "proxy"
        return tp

    data_lengths=[2 ** (n - 2) for n in range(20, 31)]
    check_iters = 10
    warmup_iters = 20
    bench_iters = 50

    # device_map = lambda rank: DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_9[rank]
    device_map = lambda rank: DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_8[rank]

    assert group.nranks == 16

    # ring1 = list(map(device_map, [0,8,9,13,12,14,15,11,10,2,3,7,6,4,5,1]))
    # ring2 = list(map(device_map, [0,1,9,8,12,13,15,14,10,11,3,2,6,7,5,4]))
    # ring3 = list(map(device_map, [0,1,10,11,15,14,13,12,8,9,2,3,7,6,5,4]))

    # k = 3
    # Ts, Cs = {}, {}
    # for i, ring in enumerate([ring1, ring2, ring3]):
    #     for u in range(16):
    #         start = ring.index(u)
    #         Ts[u, i] = [[(ring[(start + d) % 16], ring[(start + d + 1) % 16])] for d in range(15)]
    #         Cs[u, i] = 1

    k = 1
    tree_name = f"amd_sym_nnodes1_IB_13_xGMI35_bw_k_{k}_187"
    with open(f"/home/amdautomation/liangyu/mscclpp-public/amd_trees/{tree_name}.pkl", "rb") as f:
        Ts, Cs = pickle.load(f)
    
    nTs, nCs = {}, {}
    for u, i in Ts:
        nTs[device_map(u), i] = [[(device_map(a), device_map(b)) for a, b in l] for l in Ts[u, i]]
        nCs[device_map(u), i] = Cs[u, i]

    Ts, Cs = nTs, nCs

    run_allgather(Ts, Cs, k, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  data_lengths=data_lengths,
                  send_lengths=[2 ** 18],
                  check_iters=check_iters,
                  warmup_iters=warmup_iters,
                  iters=bench_iters,
                  n_parallel_sm_blocks=1)

    if group.my_rank == 0:
        print()

    run_reduce_scatter(Ts, Cs, k, group=group, connections=connections, 
                       connection_types={dest: channel_type(dest) for dest in connections},
                       data_lengths=data_lengths,
                       send_lengths=[2 ** 18],
                       scratch_size=2 ** 24,
                       check_iters=check_iters,
                       warmup_iters=warmup_iters,
                       iters=bench_iters,
                       n_parallel_sm_blocks=2,
                       n_parallel_reduce_blocks=2,
                       coll_re=True,
                       skip_leaf_tb=True,
                       n_pipeline=1)

    if group.my_rank == 0:
        print()

    run_allreduce(Ts, Cs, k, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  data_lengths=data_lengths,
                  send_lengths=[2 ** 15],
                  scratch_size=2 ** 20,
                  check_iters=check_iters,
                  warmup_iters=warmup_iters,
                  iters=bench_iters)

    if group.my_rank == 0:
        print()

    # reduce-scatter + allgather
    connections = connect_nvlink(group, [r for r in range(group.nranks) if r != group.my_rank])
    run_fusion_allreduce(Ts, Cs, k, Ts, Cs, k,
                         group=group, connections=connections, 
                         connection_types={dest: channel_type(dest) for dest in connections},
                         data_lengths=data_lengths,
                         rs_send_lengths={l: [2 ** 18] for l in data_lengths},
                         ag_send_lengths={l: [2 ** 19] for l in data_lengths},
                         scratch_size=2 ** 20,
                         check_iters=check_iters,
                         warmup_iters=warmup_iters,
                         iters=bench_iters,
                         coll_re=True,
                         skip_leaf_tb=True,
                         rs_n_parallel_sm_blocks=4,
                         ag_n_parallel_sm_blocks=1)

    del group
