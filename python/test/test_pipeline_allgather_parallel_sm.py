import cupy as cp
import pytest
import itertools

from mscclpp import ProxyService
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group
from .test_mscclpp import create_and_connect

from .pipeline_schedule import AllgatherParallelSMPipelineKernel, allgather_kernel


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_send_recv_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, sm_node_size: int,
                         n_parallel_sm_blocks: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_send_recv_chain("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    if group.my_rank == 0:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        dest = group.my_rank + 1
        if group.my_rank // sm_node_size == dest // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    elif group.my_rank == group.nranks - 1:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        src = group.my_rank - 1
        if group.my_rank // sm_node_size == src // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(memory, connections[src], src)]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]}
    else:
        # recv send
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        src = group.my_rank - 1
        dest = group.my_rank + 1
        if group.my_rank // sm_node_size == src // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(memory, connections[src], src)]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]}
        if group.my_rank // sm_node_size == dest // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    node_types = {0: 1}
    ntrees = 1

    kernel = AllgatherParallelSMPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                               memory, data_chunk_offsets, data_chunk_sizes, total_chunks, node_types,
                                               ntrees, n_parallel_sm_blocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_multipeer_broadcast(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int,
                             sm_node_size: int, n_parallel_sm_blocks: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multipeer_broadcast("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    recv_sm_channels, send_sm_channels = {}, {}
    recv_proxy_channels, send_proxy_channels = {}, {}
    if group.my_rank == 0:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        send_sm_channels = {0: list(group.make_sm_channels(memory, 
            {dest: connections[dest] for dest in range(group.my_rank + 1, group.nranks)
             if group.my_rank // sm_node_size == dest // sm_node_size}).values())}
        send_proxy_channels = {0: list(group.make_proxy_channels(proxy_service, memory, 
            {dest: connections[dest] for dest in range(group.my_rank + 1, group.nranks)
             if group.my_rank // sm_node_size != dest // sm_node_size}).values())}
    else:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        if group.my_rank // sm_node_size == 0 // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(memory, connections[0], 0)]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[0], 0)]}
    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    node_types = {0: 1}
    ntrees = 1

    kernel = AllgatherParallelSMPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels, 
                                               memory, data_chunk_offsets, data_chunk_sizes, total_chunks, node_types, ntrees)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allgather_length", [128, 2 ** 20])
@pytest.mark.parametrize("ninstance", [1, 2])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_allpair_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                           ninstance: int, sm_node_size: int, n_parallel_sm_blocks: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_allpair_allgather("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allgather_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allgather_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allgather_length", [128, 2 ** 20])
@pytest.mark.parametrize("ninstance", [1, 2])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_ring_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                        ninstance: int, sm_node_size: int, n_parallel_sm_blocks: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_ring_allgather("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allgather_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allgather_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allgather_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_tree_allgather(nelem_per_send: int, allgather_length: int, sm_node_size: int,
                        n_parallel_sm_blocks: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_allgather("
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
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

    memory = cp.array([group.my_rank + 1] * allgather_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allgather_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allgather_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("iters", [2, 10])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_multrun_allgather(nelem_per_send: int, allgather_length: int, iters: int,
                           sm_node_size: int, n_parallel_sm_blocks: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multrun_allgather("
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"iters={iters}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
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

    init_data = cp.array([group.my_rank + 1] * allgather_length, dtype=cp.int32)
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    memory = cp.zeros(allgather_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    proxy_service.start_proxy()
    for _ in range(iters):
        cp.copyto(memory, init_data)
        kernel(nelem_total=allgather_length, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()  # Causing fifo_device assertion failure
                                             # if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
@pytest.mark.parametrize("n_parallel_sm_blocks", [1, 2, 4])
def test_vary_size_allgather(nelem_per_send: int, sm_node_size: int, n_parallel_sm_blocks: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_vary_size_allreduce("
              f"nelem_per_send={nelem_per_send}, "
              f"sm_node_size={sm_node_size}, "
              f"n_parallel_sm_blocks={n_parallel_sm_blocks})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
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

    allgather_lengths = [3 * 2 ** n for n in range(10, 21)]
    max_length = max(allgather_lengths)
    init_data = cp.array([group.my_rank + 1] * max_length, dtype=cp.int32)
    memory = cp.ones(max_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              proxy_service=proxy_service,
                              n_parallel_sm_blocks=n_parallel_sm_blocks)

    funcs = [kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
             for length in allgather_lengths]
    expected = [cp.array([off // (length // group.nranks) + 1 for off in range(length)] +
                         [group.my_rank + 1] * (max_length - length), dtype=cp.int32)
                for length in allgather_lengths]

    proxy_service.start_proxy()
    for i in range(len(allgather_lengths)):
        cp.copyto(memory, init_data)
        funcs[i]()
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected[i])
    proxy_service.stop_proxy()
