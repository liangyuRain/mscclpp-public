import cupy as cp
import pytest
import itertools

from mscclpp import ProxyService
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group
from .test_mscclpp import create_and_connect

from .pipeline_schedule import PipelineKernel, allreduce_kernel, allgather_kernel


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
def test_proxy_send_recv_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    if group.my_rank == 0:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        dest = group.my_rank + 1
        recv_proxy_channels = {}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    elif group.my_rank == group.nranks - 1:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        src = group.my_rank - 1
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]}
        send_proxy_channels = {}
    else:
        # recv send
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        src = group.my_rank - 1
        dest = group.my_rank + 1
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[src], src)]}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
    data_offsets = {0: 0}
    data_sizes = {0: nelem_total}
    scratch_size = 0
    recv_proxy_scratches = {}
    node_types = {0: 1}
    nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
def test_proxy_send_recv_reduce_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    memory = cp.ones(nelem_total, dtype=cp.int32)
    if group.my_rank == 0:
        # sender
        dest = group.my_rank + 1
        recv_proxy_channels = {}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
        recv_proxy_scratches = {}
    elif group.my_rank == group.nranks - 1:
        # recver
        scratch = cp.array([1000] * scratch_size, dtype=cp.int32)
        src = group.my_rank - 1
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, scratch, connections[src], src)]}
        send_proxy_channels = {}
        recv_proxy_scratches = {0: [scratch]}
    else:
        # recv reduce send
        scratch = cp.array([1000] * scratch_size, dtype=cp.int32)
        src = group.my_rank - 1
        dest = group.my_rank + 1
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, scratch, connections[src], src)]}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)]}
        recv_proxy_scratches = {0: [scratch]}
    data_offsets = {0: 0}
    data_sizes = {0: nelem_total}
    node_types = {0: -1}
    nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert cp.array_equal(memory, cp.array([group.my_rank + 1] * nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
def test_proxy_multipeer_broadcast(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    if group.my_rank == 0:
        # sender
        memory = cp.arange(nelem_total, dtype=cp.int32)
        recv_proxy_channels = {}
        send_proxy_channels = {0: list(group.make_proxy_channels(proxy_service, memory, 
            {dest: connections[dest] for dest in range(group.my_rank + 1, group.nranks)}).values())}
    else:
        # recver
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[0], 0)]}
        send_proxy_channels = {}
    data_offsets = {0: 0}
    data_sizes = {0: nelem_total}
    scratch_size = 0
    recv_proxy_scratches = {}
    node_types = {0: 1}
    nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
def test_proxy_multipeer_reduce(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    memory = cp.ones(nelem_total, dtype=cp.int32)
    if group.my_rank == 0:
        # recver
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) 
                                    for _ in range(1, group.nranks)]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][dest - 1], connections[dest], dest)
                                   for dest in range(1, group.nranks)]}
        send_proxy_channels = {}
    else:
        # sender
        recv_proxy_scratches = {}
        recv_proxy_channels = {}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[0], 0)]}
    data_offsets = {0: 0}
    data_sizes = {0: nelem_total}
    node_types = {0: -1}
    nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    if group.my_rank == 0:
        assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
def test_proxy_root1(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    root_rank = group.nranks // 2
    if group.my_rank < root_rank:
        # reduce node
        memory = cp.ones(nelem_total, dtype=cp.int32)
        recv_proxy_scratches = {}
        recv_proxy_channels = {}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        node_types = {0: -1}
    elif group.my_rank == root_rank:
        # root
        memory = cp.ones(nelem_total, dtype=cp.int32)
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) 
                                    for _ in range(root_rank)]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][dest], connections[dest], dest)
                                   for dest in range(root_rank)]}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)
                                   for dest in range(root_rank + 1, group.nranks)]}
        node_types = {0: 0}
    else:
        # broadcast node
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        recv_proxy_scratches = {}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        send_proxy_channels = {}
        node_types = {0: 1}
    data_offsets = {0: 0}
    data_sizes = {0: nelem_total}
    nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    if group.my_rank >= root_rank:
        assert cp.array_equal(memory, cp.array([root_rank + 1] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
def test_proxy_root2(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    root_rank = group.nranks - 1
    if group.my_rank < root_rank:
        # reduce at tb0, broadcast at tb1
        memory = cp.ones(nelem_total, dtype=cp.int32)
        recv_proxy_scratches = {}
        recv_proxy_channels = {1: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        node_types = {0: -1, 1: 1}
        data_offsets = {0: 0, 1: 0}
        data_sizes = {0: nelem_total, 1: nelem_total}
        nblocks = 2
    elif group.my_rank == root_rank:
        # root
        memory = cp.ones(nelem_total, dtype=cp.int32)
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) 
                                    for _ in range(root_rank)]}
        # Creating send channels first because non-root nodes create recv channels first
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)
                                   for dest in range(root_rank)]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][dest], connections[dest], dest)
                                   for dest in range(root_rank)]}
        node_types = {0: 0}
        data_offsets = {0: 0}
        data_sizes = {0: nelem_total}
        nblocks = 1

    kernel = PipelineKernel({}, {}, recv_proxy_channels, send_proxy_channels, memory, data_offsets, 
                            data_sizes, scratch_size, {}, recv_proxy_scratches, node_types, 
                            nelem_per_send, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allreduce_length", [96, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
def test_proxy_allpair_allreduce(mpi_group: MpiGroup, nelem_per_send: int, allreduce_length: int,
                              scratch_size: int, ninstance: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              data=memory,
                              allreduce_length=allreduce_length,
                              nelem_per_send=nelem_per_send,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allreduce_length", [96, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
def test_proxy_ring_allreduce(mpi_group: MpiGroup, nelem_per_send: int, allreduce_length: int,
                              scratch_size: int, ninstance: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              data=memory,
                              allreduce_length=allreduce_length,
                              nelem_per_send=nelem_per_send,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allreduce_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
def test_proxy_tree_allreduce(nelem_per_send: int, allreduce_length: int, scratch_size: int):
    mpi_group = MpiGroup(list(range(8)))
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    Ts = {}
    Cs = {}
    l_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    r_tree = [(0, 7), (0, 6), (7, 5), (7, 4), (6, 3), (6, 2), (5, 1)]
    for u in range(group.nranks):
        Ts[u, 0] = [[((a + u) % 8, (b + u) % 8)] for a, b in l_tree]
        Ts[u, 1] = [[((a + u) % 8, (b + u) % 8)] for a, b in r_tree]
        Cs[u, 0], Cs[u, 1] = (1, 2) if u % 2 == 0 else (2, 1)

    memory = cp.array([group.my_rank + 1] * allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              data=memory,
                              allreduce_length=allreduce_length,
                              nelem_per_send=nelem_per_send,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allreduce_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("iters", [2, 10])
def test_proxy_multrun_allreduce(nelem_per_send: int, allreduce_length: int, scratch_size: int,
                              iters: int):
    mpi_group = MpiGroup(list(range(8)))
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    Ts = {}
    Cs = {}
    l_tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    r_tree = [(0, 7), (0, 6), (7, 5), (7, 4), (6, 3), (6, 2), (5, 1)]
    for u in range(group.nranks):
        Ts[u, 0] = [[((a + u) % 8, (b + u) % 8)] for a, b in l_tree]
        Ts[u, 1] = [[((a + u) % 8, (b + u) % 8)] for a, b in r_tree]
        Cs[u, 0], Cs[u, 1] = (1, 2) if u % 2 == 0 else (2, 1)

    memory = cp.ones(allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              data=memory,
                              allreduce_length=allreduce_length,
                              nelem_per_send=nelem_per_send,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    expected = [cp.array([group.nranks ** (i + 1)] * allreduce_length, dtype=cp.int32)
                for i in range(iters)]

    proxy_service.start_proxy()
    for i in range(iters):
        group.barrier()  # Add barrier here to prevent initialization overwrites the data
                         # another gpu is still getting in previous iter.
        kernel()
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected[i])
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allgather_length", [96, 2 ** 20])
@pytest.mark.parametrize("ninstance", [1, 4])
def test_proxy_allpair_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                              ninstance: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allgather_length, dtype=cp.int32)

    kernel = allgather_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              data=memory,
                              allgather_length=allgather_length,
                              nelem_per_send=nelem_per_send,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allgather_length", [96, 2 ** 20])
@pytest.mark.parametrize("ninstance", [1, 4])
def test_proxy_ring_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                           ninstance: int):
    group, connections = create_and_connect(mpi_group, "NVLink")
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
                              data=memory,
                              allgather_length=allgather_length,
                              nelem_per_send=nelem_per_send,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allgather_length", [3 * 1024, 3 * 2 ** 20])
def test_proxy_tree_allgather(nelem_per_send: int, allgather_length: int):
    mpi_group = MpiGroup(list(range(8)))
    group, connections = create_and_connect(mpi_group, "NVLink")
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
                              data=memory,
                              allgather_length=allgather_length,
                              nelem_per_send=nelem_per_send,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel()
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()
    expected = cp.array([off // (allgather_length // group.nranks) + 1
                         for off in range(allgather_length)], dtype=cp.int32)
    assert cp.array_equal(memory, expected)


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allgather_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("iters", [2, 10])
def test_proxy_multrun_allgather(nelem_per_send: int, allgather_length: int, iters: int):
    mpi_group = MpiGroup(list(range(8)))
    group, connections = create_and_connect(mpi_group, "NVLink")
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
                              data=memory,
                              allgather_length=allgather_length,
                              nelem_per_send=nelem_per_send,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    for _ in range(iters):
        cp.copyto(memory, init_data)
        cp.cuda.runtime.deviceSynchronize()
        group.barrier()  # Prevent remote kernel call from writing to memory (ProxyChannel)
                         # before `cp.copyto(memory, init_data)` is executed.
        kernel()
        cp.cuda.runtime.deviceSynchronize()  # Causing fifo_device assertion failure
                                             # if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()
