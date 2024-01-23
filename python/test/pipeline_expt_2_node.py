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
    DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_8,
    DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_9,
    print_row, 
    run_allgather, 
    run_reduce_scatter, 
    run_allreduce,
    run_fusion_allreduce,
)
from mscclpp_mpi import MpiGroup
from mscclpp import ProxyService, Transport


def device_map(rank):
    if rank < 16:
        return DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_8[rank]
    elif rank < 32:
        return DEVICE_ID_ROCM_TO_CUPY_MAP_NODE_10_9[rank]
    else:
        assert False


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


if __name__ == "__main__":
    cp.cuda.Device(MPI.COMM_WORLD.rank % 16).use()

    mpi_group = MpiGroup(list(range(32)))
    group = mscclpp_comm.CommGroup(mpi_group.comm)

    def channel_type(dest):
        tp = "sm" if dest // 16 == group.my_rank // 16 else "proxy"
        return tp

    remote_nghrs = [v for v in range(group.nranks) if v != group.my_rank]
    connections = group.make_connection(remote_nghrs,
                                        {v: Transport.CudaIpc if v // 16 == group.my_rank // 16
                                            else group.my_ib_device(group.my_rank % 16)
                                         for v in remote_nghrs})

    data_lengths = [2 ** (n - 2) for n in range(10, 34)]
    send_lengths = [2 ** n for n in range(14, 21)]
    check_iters = 0
    warmup_iters = 100
    bench_iters = 100

    assert group.nranks == 32

    k = 1
    tree_name = f"amd_sym_nnodes2_IB_13_xGMI35_bw_k_{k}_224"
    if group.my_rank == 0:
        print(f"tree_file={tree_name}")
    with open(f"/home/amdautomation/liangyu/mscclpp-public/amd_trees/{tree_name}.pkl", "rb") as f:
        Ts, Cs = pickle.load(f)
    
    nTs, nCs = {}, {}
    for u, i in Ts:
        nTs[device_map(u), i] = [[(device_map(a), device_map(b)) for a, b in l] for l in Ts[u, i]]
        nCs[device_map(u), i] = Cs[u, i]

    Ts, Cs = nTs, nCs

    # Allgather
    for ninstance in [1, 2, 3, 4, 6, 8]:
        max_sm_blocks = math.floor((MAX_NBLOCKS / (ninstance * k) - 2) / 14)
        if max_sm_blocks <= 0:
            break
        nsm = [1, 2, 3, 4, 6, 8]
        nsm = [v for v in nsm if v < max_sm_blocks] + [max_sm_blocks]
        for n_parallel_sm_blocks in nsm:
            Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
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

    # ReduceScatter
    for ninstance in [1, 2, 3, 4, 6, 8]:
        max_sm_blocks = math.floor((MAX_NBLOCKS / (ninstance * k) - 2) / 14)
        if max_sm_blocks <= 0:
            break
        nsm = [1, 2, 3, 4, 6, 8]
        nsm = [v for v in nsm if v < max_sm_blocks] + [max_sm_blocks]
        for n_parallel_sm_blocks in nsm:
            max_reduce_blocks = math.floor((MAX_NBLOCKS / (ninstance * k) - n_parallel_sm_blocks * 14) / 2)
            if max_reduce_blocks <= 0:
                break
            nreduce = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32]
            nreduce = [v for v in nreduce if v < max_reduce_blocks] + [max_reduce_blocks]
            for n_parallel_reduce_blocks in nreduce:
                max_n_pipeline = math.floor(MAX_NBLOCKS / (ninstance * k) / (n_parallel_reduce_blocks * 2 + n_parallel_sm_blocks * 14))
                if max_n_pipeline <= 0:
                    break
                npipeline = [1, 2, 3, 4]
                # npipeline = [v for v in npipeline if v < max_n_pipeline] + [max_n_pipeline]
                for n_pipeline in npipeline:
                    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
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
                                        n_parallel_reduce_blocks=n_parallel_reduce_blocks,
                                        #    sendtb=True,
                                        n_pipeline=n_pipeline,
                                        coll_re=True,
                                        skip_leaf_tb=True)
                    except ThreadBlockLimitException as e:
                        # Exception may not be triggered at all ranks.
                        # Different ranks may requre different num of threadblocks depending on parameters.
                        print(f"ThreadBlockLimitException: "
                            f"nblocks={e.nblocks}, ninstance={ninstance}, "
                            f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
                            f"n_parallel_reduce_blocks={n_parallel_reduce_blocks}")
                    if group.my_rank == 0:
                        print()

    # Allreduce
    # mscclpp_path = "/root/mscclpp-public"
    # with open(f"{mscclpp_path}/run_configs/reduceScatter_best_config_coll_re.pkl", "rb") as f:
    #     reduceScatter_configs_coll_re = pickle.load(f)
    # with open(f"{mscclpp_path}/run_configs/reduceScatter_best_config_sendtb.pkl", "rb") as f:
    #     reduceScatter_configs_sendtb = pickle.load(f)
    # with open(f"{mscclpp_path}/run_configs/allgather_best_config.pkl", "rb") as f:
    #     allgather_configs = pickle.load(f)

    # for length in data_lengths:
    #     rs_config = reduceScatter_configs_coll_re[length * 4]
    #     coll_re = True
    #     sendtb = False
        
    #     # rs_config = reduceScatter_configs_sendtb[length * 4]
    #     # coll_re = False
    #     # sendtb = True
        
    #     ag_config = allgather_configs[length * 4]

    #     RS_Tsp, RS_Csp, RS_kp = multi_instance(RS_Ts, RS_Cs, RS_k, rs_config['k'])
    #     AG_Tsp, AG_Csp, AG_kp = multi_instance(AG_Ts, AG_Cs, AG_k, ag_config['k'])

    #     rs_send_size = rs_config['send_size'] // 4
    #     ag_send_size = ag_config['send_size'] // 4

    #     rs_n_parallel_sm_blocks = rs_config['n_parallel_sm_blocks']
    #     ag_n_parallel_sm_blocks = ag_config['n_parallel_sm_blocks']

    #     rs_n_parallel_reduce_blocks = rs_config['n_parallel_reduce_blocks'] if 'n_parallel_reduce_blocks' in rs_config else None
    #     assert sendtb == (rs_n_parallel_reduce_blocks is not None)

    #     scratch_size = 2 ** 24
    #     run_fusion_allreduce(RS_Tsp, RS_Csp, RS_kp, AG_Tsp, AG_Csp, AG_kp,
    #                          group=group, connections=connections, 
    #                          connection_types={dest: channel_type(dest) for dest in connections},
    #                          data_lengths=list(filter(lambda x: x <= length * 2, data_lengths)),
    #                          rs_send_lengths={l: list(filter(lambda x: x <= scratch_size, [rs_send_size // 2, rs_send_size, rs_send_size * 2])) for l in data_lengths},
    #                          ag_send_lengths={l: list(filter(lambda x: x <= scratch_size, [ag_send_size // 2, ag_send_size, ag_send_size * 2])) for l in data_lengths},
    #                          scratch_size=scratch_size,
    #                          check_iters=check_iters,
    #                          warmup_iters=warmup_iters,
    #                          iters=bench_iters,
    #                          coll_re=coll_re,
    #                          sendtb=sendtb,
    #                          skip_leaf_tb=True,
    #                          rs_n_parallel_sm_blocks=rs_n_parallel_sm_blocks,
    #                          rs_n_parallel_reduce_blocks=rs_n_parallel_reduce_blocks,
    #                          ag_n_parallel_sm_blocks=ag_n_parallel_sm_blocks)
    #     if group.my_rank == 0:
    #         print()

    del group
