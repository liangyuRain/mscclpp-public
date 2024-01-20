import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import pickle
import copy
from mpi4py import MPI
import os
import math
import pickle

import mscclpp.comm as mscclpp_comm
from pipeline_schedule import (
    MAX_NBLOCKS,
    PipelineKernel,
    ThreadBlockLimitException,
)
from pipeline_expt import (
    print_row, 
    run_allgather, 
    run_reduce_scatter, 
    run_allreduce,
    run_fusion_allreduce,
)
from mscclpp_mpi import MpiGroup
from mscclpp import ProxyService, Transport


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

    data_lengths = [3 * 2 ** (n - 2) for n in range(10, 31)]
    send_lengths = [2 ** n for n in range(14, 21)]
    check_iters = 0
    warmup_iters = 100
    bench_iters = 100

    AG_k = 1
    AG_tree_name = f"symmetric/sym_split2_IB20_NV300_bw_k_{AG_k}_320"
    if group.my_rank == 0:
        print(f"AG_tree_file={AG_tree_name}")
    with open(f"/root/mscclpp-public/trees/{AG_tree_name}.pkl", "rb") as f:
        AG_Ts, AG_Cs = pickle.load(f)

    # Allgather
    for ninstance in [1, 2, 3, 4, 6, 8]:
        assert AG_tree_name == f"symmetric/sym_split2_IB20_NV300_bw_k_{AG_k}_320"
        max_sm_blocks = math.floor((MAX_NBLOCKS / (ninstance * AG_k) - 2) / 14)
        if max_sm_blocks <= 0:
            break
        nsm = [1, 2, 3, 4, 6, 8]
        nsm = [v for v in nsm if v < max_sm_blocks] + [max_sm_blocks]
        for n_parallel_sm_blocks in nsm:
            Tsp, Csp, kp = multi_instance(AG_Ts, AG_Cs, AG_k, ninstance)
            try:
                run_allgather(Tsp, Csp, kp, group=group, connections=connections, 
                              connection_types={dest: channel_type(dest) for dest in connections},
                              data_lengths=data_lengths,
                              send_lengths=send_lengths,
                              check_iters=check_iters,
                              warmup_iters=warmup_iters,
                              iters=bench_iters,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)
            except ThreadBlockLimitException as e:
                # Exception may not be triggered at all ranks.
                # Different ranks may requre different num of threadblocks depending on parameters.
                print(f"ThreadBlockLimitException: "
                      f"nblocks={e.nblocks}, ninstance={ninstance}, "
                      f"n_parallel_sm_blocks={n_parallel_sm_blocks}")
            if group.my_rank == 0:
                print()
    
    RS_k = 1
    RS_tree_name = f"symmetric/sym_split2_IB20_NV300_bw_k_{RS_k}_320"
    if group.my_rank == 0:
        print(f"RS_tree_file={RS_tree_name}")
    with open(f"/root/mscclpp-public/trees/{RS_tree_name}.pkl", "rb") as f:
        RS_Ts, RS_Cs = pickle.load(f)

    # ReduceScatter
    for ninstance in [1, 2, 3, 4, 6, 8]:
        assert RS_tree_name == f"symmetric/sym_split2_IB20_NV300_bw_k_{RS_k}_320"
        max_sm_blocks = math.floor((MAX_NBLOCKS / (ninstance * RS_k) - 2) / 14)
        if max_sm_blocks <= 0:
            break
        nsm = [1, 2, 3, 4, 6, 8]
        nsm = [v for v in nsm if v < max_sm_blocks] + [max_sm_blocks]
        for n_parallel_sm_blocks in nsm:
            # max_reduce_blocks = math.floor((MAX_NBLOCKS / (ninstance * RS_k) - n_parallel_sm_blocks * 14) / 2)
            # if max_reduce_blocks <= 0:
            #     break
            # nreduce = [1, 2, 4, 8, 16, 32]
            # nreduce = [v for v in nreduce if v < max_reduce_blocks] + [max_reduce_blocks]
            # for n_parallel_reduce_blocks in nreduce:
            #     Tsp, Csp, kp = multi_instance(RS_Ts, RS_Cs, RS_k, ninstance)
            #     try:
            #         run_reduce_scatter(Tsp, Csp, kp, group=group, connections=connections, 
            #                            connection_types={dest: channel_type(dest) for dest in connections},
            #                            data_lengths=data_lengths,
            #                            send_lengths=send_lengths,
            #                            scratch_size=2 ** 24,
            #                            check_iters=check_iters,
            #                            warmup_iters=warmup_iters,
            #                            iters=bench_iters,
            #                            n_parallel_sm_blocks=n_parallel_sm_blocks,
            #                            n_parallel_reduce_blocks=n_parallel_reduce_blocks,
            #                            skip_leaf_tb=True,
            #                            sendtb=True)
            #     except ThreadBlockLimitException as e:
            #         # Exception may not be triggered at all ranks.
            #         # Different ranks may requre different num of threadblocks depending on parameters.
            #         print(f"ThreadBlockLimitException: "
            #               f"nblocks={e.nblocks}, ninstance={ninstance}, "
            #               f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
            #               f"n_parallel_reduce_blocks={n_parallel_reduce_blocks}")
            #     if group.my_rank == 0:
            #         print()
            Tsp, Csp, kp = multi_instance(RS_Ts, RS_Cs, RS_k, ninstance)
            try:
                run_reduce_scatter(Tsp, Csp, kp, group=group, connections=connections, 
                                   connection_types={dest: channel_type(dest) for dest in connections},
                                   data_lengths=data_lengths,
                                   send_lengths=send_lengths,
                                   scratch_size=2 ** 24,
                                   check_iters=check_iters,
                                   warmup_iters=warmup_iters,
                                   iters=bench_iters,
                                   n_parallel_sm_blocks=n_parallel_sm_blocks,
                                   coll_re=True,
                                   skip_leaf_tb=True)
            except ThreadBlockLimitException as e:
                # Exception may not be triggered at all ranks.
                # Different ranks may requre different num of threadblocks depending on parameters.
                print(f"ThreadBlockLimitException: "
                      f"nblocks={e.nblocks}, ninstance={ninstance}, "
                      f"n_parallel_sm_blocks={n_parallel_sm_blocks}")
            if group.my_rank == 0:
                print()

    # Allreduce
    mscclpp_path = "/root/mscclpp-public"
    with open(f"{mscclpp_path}/run_configs/reduceScatter_best_config_coll_re.pkl", "rb") as f:
        reduceScatter_configs_coll_re = pickle.load(f)
    with open(f"{mscclpp_path}/run_configs/reduceScatter_best_config_sendtb.pkl", "rb") as f:
        reduceScatter_configs_sendtb = pickle.load(f)
    with open(f"{mscclpp_path}/run_configs/allgather_best_config.pkl", "rb") as f:
        allgather_configs = pickle.load(f)

    for length in data_lengths:
        rs_config = reduceScatter_configs_coll_re[length * 4]
        coll_re = True
        sendtb = False
        
        # rs_config = reduceScatter_configs_sendtb[length * 4]
        # coll_re = False
        # sendtb = True
        
        ag_config = allgather_configs[length * 4]

        RS_Tsp, RS_Csp, RS_kp = multi_instance(RS_Ts, RS_Cs, RS_k, rs_config['k'])
        AG_Tsp, AG_Csp, AG_kp = multi_instance(AG_Ts, AG_Cs, AG_k, ag_config['k'])

        rs_send_size = rs_config['send_size'] // 4
        ag_send_size = ag_config['send_size'] // 4

        rs_n_parallel_sm_blocks = rs_config['n_parallel_sm_blocks']
        ag_n_parallel_sm_blocks = ag_config['n_parallel_sm_blocks']

        rs_n_parallel_reduce_blocks = rs_config['n_parallel_reduce_blocks'] if 'n_parallel_reduce_blocks' in rs_config else None
        assert sendtb == (rs_n_parallel_reduce_blocks is not None)

        scratch_size = 2 ** 24
        run_fusion_allreduce(RS_Tsp, RS_Csp, RS_kp, AG_Tsp, AG_Csp, AG_kp,
                             group=group, connections=connections, 
                             connection_types={dest: channel_type(dest) for dest in connections},
                             data_lengths=list(filter(lambda x: x <= length * 2, data_lengths)),
                             rs_send_lengths={l: list(filter(lambda x: x <= scratch_size, [rs_send_size // 2, rs_send_size, rs_send_size * 2])) for l in data_lengths},
                             ag_send_lengths={l: list(filter(lambda x: x <= scratch_size, [ag_send_size // 2, ag_send_size, ag_send_size * 2])) for l in data_lengths},
                             scratch_size=scratch_size,
                             check_iters=check_iters,
                             warmup_iters=warmup_iters,
                             iters=bench_iters,
                             coll_re=coll_re,
                             sendtb=sendtb,
                             skip_leaf_tb=True,
                             rs_n_parallel_sm_blocks=rs_n_parallel_sm_blocks,
                             rs_n_parallel_reduce_blocks=rs_n_parallel_reduce_blocks,
                             ag_n_parallel_sm_blocks=ag_n_parallel_sm_blocks)
        if group.my_rank == 0:
            print()

    del group
