import cupy as cp
from cupyx.profiler import benchmark
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
from pipeline_expt import (
    print_row, 
    run_allgather, 
    run_reduce_scatter, 
    run_allreduce,
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

    k = 1
    tree_name = f"symmetric/sym_split_IB20_NV300_bw_k_{k}_320"
    if group.my_rank == 0:
        print(f"tree_file={tree_name}")
    with open(f"/root/mscclpp-public/trees/{tree_name}.pkl", "rb") as f:
        Ts, Cs = pickle.load(f)

    data_lengths=[2 ** (n - 2) for n in range(10, 31)]

    # Allgather
    ninstance = 1
    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
    run_allgather(Tsp, Csp, kp, group=group, connections=connections, 
                  connection_types={dest: channel_type(dest) for dest in connections},
                  data_lengths=data_lengths,
                  nelem_per_send=[2 ** 18],
                  warmup_iters=20,
                  iters=50)

    if group.my_rank == 0:
        print()

    # ReduceScatter
    ninstance = 1
    Tsp, Csp, kp = multi_instance(Ts, Cs, k, ninstance)
    run_reduce_scatter(Tsp, Csp, kp, group=group, connections=connections, 
                       connection_types={dest: channel_type(dest) for dest in connections},
                       data_lengths=data_lengths,
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
                  data_lengths=data_lengths,
                  nelem_per_send=2 ** 15,
                  scratch_size=2 ** 20,
                  warmup_iters=20,
                  iters=50)

    del group
