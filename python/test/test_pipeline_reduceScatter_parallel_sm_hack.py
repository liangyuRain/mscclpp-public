import cupy as cp
import pytest
import itertools

from mscclpp import ProxyService
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group
from .test_mscclpp import create_and_connect

from .pipeline_schedule import (
    ReduceScatterParallelSMHackPipelineKernel,
    reduce_scatter_kernel_hack,
)


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_send_recv_reduce_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int,
                                scratch_size: int, sm_node_size: int, n_parallel_sm_blocks: int,
                                skip_leaf_tb: bool):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_send_recv_reduce_chain("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    memory = cp.ones(nelem_total, dtype=cp.int32)
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}
    if group.my_rank == 0:
        # sender
        dest = group.my_rank + 1
        src = None
        if group.my_rank // sm_node_size == dest // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    elif group.my_rank == group.nranks - 1:
        # recver
        scratch = cp.array([1000] * scratch_size, dtype=cp.int32)
        src = group.my_rank - 1
        if group.my_rank // sm_node_size == src // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(scratch, connections[src], src)]}
            recv_sm_scratches = {0: [scratch]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, scratch, connections[src], src)]}
            recv_proxy_scratches = {0: [scratch]}
    else:
        # recv reduce send
        scratch = cp.array([1000] * scratch_size, dtype=cp.int32)
        src = group.my_rank - 1
        dest = group.my_rank + 1
        if group.my_rank // sm_node_size == src // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(scratch, connections[src], src)]}
            recv_sm_scratches = {0: [scratch]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, scratch, connections[src], src)]}
            recv_proxy_scratches = {0: [scratch]}
        if group.my_rank // sm_node_size == dest // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    ntrees = 1

    if group.my_rank == 0 and skip_leaf_tb and group.my_rank // sm_node_size == dest // sm_node_size:
        proxy_service.start_proxy()
        group.barrier()
    else:
        if skip_leaf_tb:
            leaf_nodes = {0: [] if src is None else [src == 0]}
        else:
            leaf_nodes = None
        kernel = ReduceScatterParallelSMHackPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                                           memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                                                           recv_sm_scratches, recv_proxy_scratches, ntrees, n_parallel_sm_blocks,
                                                           leaf_nodes, skip_leaf_tb)
        proxy_service.start_proxy()
        group.barrier()
        kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()

    if skip_leaf_tb:
        group.barrier()
    assert cp.array_equal(memory, cp.array([group.my_rank + 1] * nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(3)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_multipeer_reduce(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int,
                          n_parallel_sm_blocks: int, skip_leaf_tb: bool):
    sm_node_size = 2
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multipeer_reduce("
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    assert group.nranks == 3
    proxy_service = ProxyService()

    memory = cp.ones(nelem_total, dtype=cp.int32)
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}
    if group.my_rank == 0:
        # recver
        sm_recv_peers = [dest for dest in range(1, group.nranks) if group.my_rank // sm_node_size == dest // sm_node_size]
        recv_sm_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in sm_recv_peers]}
        recv_sm_channels = {0: [group.make_sm_channel(recv_sm_scratches[0][idx], connections[dest], dest)
                                for idx, dest in enumerate(sm_recv_peers)]}
        
        proxy_recv_peers = [dest for dest in range(1, group.nranks) if group.my_rank // sm_node_size != dest // sm_node_size]
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in proxy_recv_peers]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][idx], connections[dest], dest)
                                   for idx, dest in enumerate(proxy_recv_peers)]}
    else:
        # sender
        if group.my_rank // sm_node_size == 0 // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[0], 0)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[0], 0)]}
    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    ntrees = 1

    if group.my_rank != 0 and skip_leaf_tb and group.my_rank // sm_node_size == 0 // sm_node_size:
        proxy_service.start_proxy()
        group.barrier()
    else:
        if skip_leaf_tb:
            if group.my_rank == 0:
                leaf_nodes = {0: [True for _ in range(1, group.nranks)]}
            else:
                leaf_nodes = {0: []}
        else:
            leaf_nodes = None
        kernel = ReduceScatterParallelSMHackPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                                           memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                                                           recv_sm_scratches, recv_proxy_scratches, ntrees, n_parallel_sm_blocks,
                                                           leaf_nodes, skip_leaf_tb)
        proxy_service.start_proxy()
        group.barrier()
        kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()

    if skip_leaf_tb:
        group.barrier()
    if group.my_rank == 0:
        assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_tree_reduce_to_root(nelem_per_send: int, nelem_total: int, scratch_size: int,
                             n_parallel_sm_blocks: int, skip_leaf_tb: bool):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_reduce_to_root("
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    recv_peers = [src for dest, src in tree if dest == group.my_rank]
    send_peers = [dest for dest, src in tree if src == group.my_rank]
    assert len(send_peers) <= 1

    memory = cp.ones(nelem_total, dtype=cp.int32)
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}

    sm_recv_peers = [dest for dest in recv_peers if group.my_rank % 2 == dest % 2]
    sm_send_peers = [dest for dest in send_peers if group.my_rank % 2 == dest % 2]
    if len(sm_recv_peers) > 0:
        recv_sm_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in sm_recv_peers]}
        recv_sm_channels = {0: [group.make_sm_channel(recv_sm_scratches[0][idx], connections[dest], dest)
                                for idx, dest in enumerate(sm_recv_peers)]}
    if len(sm_send_peers) > 0:
        send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)
                                for dest in sm_send_peers]}
    
    proxy_recv_peers = [dest for dest in recv_peers if group.my_rank % 2 != dest % 2]
    proxy_send_peers = [dest for dest in send_peers if group.my_rank % 2 != dest % 2]
    if len(proxy_recv_peers) > 0:
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in proxy_recv_peers]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][idx], connections[dest], dest)
                                   for idx, dest in enumerate(proxy_recv_peers)]}
    if len(proxy_send_peers) > 0:
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)
                                   for dest in proxy_send_peers]}

    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    ntrees = 1

    if group.my_rank in [4, 5, 6, 7] and skip_leaf_tb and group.my_rank % 2 == send_peers[0] % 2:
        proxy_service.start_proxy()
        group.barrier()
    else:
        if skip_leaf_tb:
            leaf_nodes = {0: [src in [4, 5, 6, 7] for src in recv_peers]}
        else:
            leaf_nodes = None
        kernel = ReduceScatterParallelSMHackPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                                           memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                                                           recv_sm_scratches, recv_proxy_scratches, ntrees, n_parallel_sm_blocks,
                                                           leaf_nodes, skip_leaf_tb)
        proxy_service.start_proxy()
        group.barrier()
        kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()

    if skip_leaf_tb:
        group.barrier()
    expected_res = {
        0: 8, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1
    }
    assert expected_res[group.my_rank] == 1 + sum(expected_res[src] for src in recv_peers)
    assert cp.array_equal(memory, cp.array([expected_res[group.my_rank]] * nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("reduce_scatter_length", [128, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 2])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_ring_reduce_scatter(mpi_group: MpiGroup, nelem_per_send: int, reduce_scatter_length: int,
                             scratch_size: int, ninstance: int, sm_node_size: int,
                             n_parallel_sm_blocks: int, skip_leaf_tb: bool):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_ring_reduce_scatter("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel_hack(Ts, Cs,
                                        k=ninstance,
                                        group=group,
                                        connections=connections,
                                        connection_types=connection_types,
                                        data=memory,
                                        scratch_size=scratch_size,
                                        proxy_service=proxy_service,
                                        n_parallel_sm_blocks=n_parallel_sm_blocks,
                                        skip_leaf_tb=skip_leaf_tb)

    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    if skip_leaf_tb:
        group.barrier()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
    assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("reduce_scatter_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_tree_reduce_scatter(nelem_per_send: int, reduce_scatter_length: int, scratch_size: int,
                             n_parallel_sm_blocks: int, skip_leaf_tb: bool):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank % 2 == dest % 2 else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {}
    Cs = {}
    l_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    r_tree = [(0, 7), (0, 6), (7, 5), (7, 4), (6, 3), (6, 2), (5, 1)]
    for u in range(group.nranks):
        Ts[u, 0] = [[((a + u) % 8, (b + u) % 8)] for a, b in l_tree]
        Ts[u, 1] = [[((a + u) % 8, (b + u) % 8)] for a, b in r_tree]
        Cs[u, 0], Cs[u, 1] = (1, 2) if u % 2 == 0 else (2, 1)

    memory = cp.array([group.my_rank + 1] * reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel_hack(Ts, Cs,
                                        k=3,
                                        group=group,
                                        connections=connections,
                                        connection_types=connection_types,
                                        data=memory,
                                        scratch_size=scratch_size,
                                        proxy_service=proxy_service,
                                        n_parallel_sm_blocks=n_parallel_sm_blocks,
                                        skip_leaf_tb=skip_leaf_tb)
    
    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    if skip_leaf_tb:
        group.barrier()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
    assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("reduce_scatter_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("iters", [2, 10])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_multrun_reduce_scatter(nelem_per_send: int, reduce_scatter_length: int, scratch_size: int,
                                iters: int, n_parallel_sm_blocks: int, skip_leaf_tb: bool):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multrun_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"iters={iters}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank % 2 == dest % 2 else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {}
    Cs = {}
    l_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    r_tree = [(0, 7), (0, 6), (7, 5), (7, 4), (6, 3), (6, 2), (5, 1)]
    for u in range(group.nranks):
        Ts[u, 0] = [[((a + u) % 8, (b + u) % 8)] for a, b in l_tree]
        Ts[u, 1] = [[((a + u) % 8, (b + u) % 8)] for a, b in r_tree]
        Cs[u, 0], Cs[u, 1] = (1, 2) if u % 2 == 0 else (2, 1)

    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    init_data = cp.ones(reduce_scatter_length, dtype=cp.int32)
    expected = cp.array([group.nranks] * shard_size, dtype=cp.int32)
    memory = cp.zeros(reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel_hack(Ts, Cs,
                                        k=3,
                                        group=group,
                                        connections=connections,
                                        connection_types=connection_types,
                                        data=memory,
                                        scratch_size=scratch_size,
                                        proxy_service=proxy_service,
                                        n_parallel_sm_blocks=n_parallel_sm_blocks,
                                        skip_leaf_tb=skip_leaf_tb)

    proxy_service.start_proxy()
    for _ in range(iters):
        cp.copyto(memory, init_data)
        if skip_leaf_tb:
            cp.cuda.runtime.deviceSynchronize()
            group.barrier()
        kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        if skip_leaf_tb:
            group.barrier()
        assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
@pytest.mark.parametrize("skip_leaf_tb", [False, True])
def test_vary_size_reduce_scatter(nelem_per_send: int, scratch_size: int, n_parallel_sm_blocks: int,
                                  skip_leaf_tb: bool):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_vary_size_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"scratch_size={scratch_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks}, "
              f"skip_leaf_tb={skip_leaf_tb})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank % 2 == dest % 2 else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {}
    Cs = {}
    l_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    r_tree = [(0, 7), (0, 6), (7, 5), (7, 4), (6, 3), (6, 2), (5, 1)]
    for u in range(group.nranks):
        Ts[u, 0] = [[((a + u) % 8, (b + u) % 8)] for a, b in l_tree]
        Ts[u, 1] = [[((a + u) % 8, (b + u) % 8)] for a, b in r_tree]
        Cs[u, 0], Cs[u, 1] = (1, 2) if u % 2 == 0 else (2, 1)

    reduce_scatter_lengths = [3 * 2 ** n for n in range(10, 21)]
    max_length = max(reduce_scatter_lengths)
    init_data = cp.ones(max_length, dtype=cp.int32)
    memory = cp.ones(max_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel_hack(Ts, Cs,
                                        k=3,
                                        group=group,
                                        connections=connections,
                                        connection_types=connection_types,
                                        data=memory,
                                        scratch_size=scratch_size,
                                        proxy_service=proxy_service,
                                        n_parallel_sm_blocks=n_parallel_sm_blocks,
                                        skip_leaf_tb=skip_leaf_tb)

    funcs = [kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
             for length in reduce_scatter_lengths]
    shard_sizes = [length // group.nranks for length in reduce_scatter_lengths]
    expected = [cp.array([group.nranks] * shard_size, dtype=cp.int32) for shard_size in shard_sizes]
    shard_begins = [shard_size * group.my_rank for shard_size in shard_sizes]
    shard_ends = [shard_begin + shard_size for shard_begin, shard_size in zip(shard_begins, shard_sizes)]

    proxy_service.start_proxy()
    for i in range(len(reduce_scatter_lengths)):
        cp.copyto(memory, init_data)
        if skip_leaf_tb:
            cp.cuda.runtime.deviceSynchronize()
            group.barrier()
        funcs[i]()
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        if skip_leaf_tb:
            group.barrier()
        assert cp.array_equal(memory[shard_begins[i]: shard_ends[i]], expected[i])
    proxy_service.stop_proxy()
