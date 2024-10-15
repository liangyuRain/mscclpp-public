import cupy as cp
import pytest
import itertools

from mscclpp import ProxyService
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group
from .test_mscclpp import create_and_connect

from .pipeline_schedule import PipelineKernel, allreduce_kernel, allgather_kernel, reduce_scatter_kernel


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_send_recv_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_send_recv_chain("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
    scratch_size = 0
    node_types = {0: 1}
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            {}, {}, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_send_recv_reduce_chain(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int,
                                scratch_size: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_send_recv_reduce_chain("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    memory = cp.ones(nelem_total, dtype=cp.int32)
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}
    if group.my_rank == 0:
        # sender
        dest = group.my_rank + 1
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
    node_types = {0: -1}
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            recv_sm_scratches, recv_proxy_scratches, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.array([group.my_rank + 1] * nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_multipeer_broadcast(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int,
                             sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multipeer_broadcast("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
    scratch_size = 0
    node_types = {0: 1}
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels, 
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            {}, {}, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.arange(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_multipeer_reduce(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int,
                          sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multipeer_reduce("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
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
    node_types = {0: -1}
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            recv_sm_scratches, recv_proxy_scratches, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    if group.my_rank == 0:
        assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_tree_reduce_to_root(nelem_per_send: int, nelem_total: int, scratch_size: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_reduce_to_root("
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    tree = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    recv_peers = [src for dest, src in tree if dest == group.my_rank]
    send_peers = [dest for dest, src in tree if src == group.my_rank]

    memory = cp.ones(nelem_total, dtype=cp.int32)
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}

    sm_recv_peers = [dest for dest in recv_peers if group.my_rank // sm_node_size == dest // sm_node_size]
    sm_send_peers = [dest for dest in send_peers if group.my_rank // sm_node_size == dest // sm_node_size]
    if len(sm_recv_peers) > 0:
        recv_sm_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in sm_recv_peers]}
        recv_sm_channels = {0: [group.make_sm_channel(recv_sm_scratches[0][idx], connections[dest], dest)
                                for idx, dest in enumerate(sm_recv_peers)]}
    if len(sm_send_peers) > 0:
        send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)
                                for dest in sm_send_peers]}
    
    proxy_recv_peers = [dest for dest in recv_peers if group.my_rank // sm_node_size != dest // sm_node_size]
    proxy_send_peers = [dest for dest in send_peers if group.my_rank // sm_node_size != dest // sm_node_size]
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
    node_types = {0: -1}
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            recv_sm_scratches, recv_proxy_scratches, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected_res = {
        0: 8, 1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1
    }
    assert expected_res[group.my_rank] == 1 + sum(expected_res[src] for src in recv_peers)
    assert cp.array_equal(memory, cp.array([expected_res[group.my_rank]] * nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(3, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_root1(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_root1("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"nelem_total={nelem_total}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    root_rank = group.nranks // 2
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}
    if group.my_rank < root_rank:
        # reduce node
        memory = cp.ones(nelem_total, dtype=cp.int32)
        if group.my_rank // sm_node_size == root_rank // sm_node_size:
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[root_rank], root_rank)]}
        else:
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        node_types = {0: -1}
    elif group.my_rank == root_rank:
        # root
        memory = cp.ones(nelem_total, dtype=cp.int32)

        sm_recv_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size == dest // sm_node_size]
        sm_send_peers = [dest for dest in range(root_rank + 1, group.nranks) if group.my_rank // sm_node_size == dest // sm_node_size]
        recv_sm_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in sm_recv_peers]}
        recv_sm_channels = {0: [group.make_sm_channel(recv_sm_scratches[0][idx], connections[dest], dest)
                                for idx, dest in enumerate(sm_recv_peers)]}
        send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)
                                for dest in sm_send_peers]}
        
        proxy_recv_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size != dest // sm_node_size]
        proxy_send_peers = [dest for dest in range(root_rank + 1, group.nranks) if group.my_rank // sm_node_size != dest // sm_node_size]
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in proxy_recv_peers]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][idx], connections[dest], dest)
                                   for idx, dest in enumerate(proxy_recv_peers)]}
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)
                                   for dest in proxy_send_peers]}
        node_types = {0: 0}
    else:
        # broadcast node
        memory = cp.zeros(nelem_total, dtype=cp.int32)
        if group.my_rank // sm_node_size == root_rank // sm_node_size:
            recv_sm_channels = {0: [group.make_sm_channel(memory, connections[root_rank], root_rank)]}
        else:
            recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        node_types = {0: 1}
    data_chunk_offsets = {0: 0}
    data_chunk_sizes = {0: 1}
    total_chunks = 1
    nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            recv_sm_scratches, recv_proxy_scratches, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    if group.my_rank >= root_rank:
        assert cp.array_equal(memory, cp.array([root_rank + 1] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("nelem_total", [20, 1024])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_root2(mpi_group: MpiGroup, nelem_per_send: int, nelem_total: int, scratch_size: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_root2("
                f"mpi_group.comm.size={mpi_group.comm.size}, "
                f"nelem_per_send={nelem_per_send}, "
                f"nelem_total={nelem_total}, "
                f"scratch_size={scratch_size}, "
                f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    proxy_service = ProxyService()

    root_rank = group.nranks - 1
    recv_sm_channels, send_sm_channels, recv_sm_scratches = {}, {}, {}
    recv_proxy_channels, send_proxy_channels, recv_proxy_scratches = {}, {}, {}
    if group.my_rank < root_rank:
        # reduce at tb0, broadcast at tb1
        memory = cp.ones(nelem_total, dtype=cp.int32)
        if group.my_rank // sm_node_size == root_rank // sm_node_size:
            recv_sm_channels = {1: [group.make_sm_channel(memory, connections[root_rank], root_rank)]}
            send_sm_channels = {0: [group.make_sm_channel(memory, connections[root_rank], root_rank)]}
        else:
            recv_proxy_channels = {1: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
            send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[root_rank], root_rank)]}
        node_types = {0: -1, 1: 1}
        data_chunk_offsets = {0: 0, 1: 0}
        data_chunk_sizes = {0: 1, 1: 1}
        total_chunks = 1
        nblocks = 2
    elif group.my_rank == root_rank:
        # root
        memory = cp.ones(nelem_total, dtype=cp.int32)

        sm_recv_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size == dest // sm_node_size]
        sm_send_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size == dest // sm_node_size]
        # Creating send channels first because non-root nodes create recv channels first
        send_sm_channels = {0: [group.make_sm_channel(memory, connections[dest], dest)
                                for dest in sm_send_peers]}
        recv_sm_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in sm_recv_peers]}
        recv_sm_channels = {0: [group.make_sm_channel(recv_sm_scratches[0][idx], connections[dest], dest)
                                for idx, dest in enumerate(sm_recv_peers)]}
        
        proxy_recv_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size != dest // sm_node_size]
        proxy_send_peers = [dest for dest in range(root_rank) if group.my_rank // sm_node_size != dest // sm_node_size]
        send_proxy_channels = {0: [group.make_proxy_channel(proxy_service, memory, connections[dest], dest)
                                   for dest in proxy_send_peers]}
        recv_proxy_scratches = {0: [cp.array([1000] * scratch_size, dtype=cp.int32) for _ in proxy_recv_peers]}
        recv_proxy_channels = {0: [group.make_proxy_channel(proxy_service, recv_proxy_scratches[0][idx], connections[dest], dest)
                                   for idx, dest in enumerate(proxy_recv_peers)]}

        node_types = {0: 0}
        data_chunk_offsets = {0: 0}
        data_chunk_sizes = {0: 1}
        total_chunks = 1
        nblocks = 1

    kernel = PipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                            memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                            recv_sm_scratches, recv_proxy_scratches, node_types, nblocks)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allreduce_length", [128, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_allpair_allreduce(mpi_group: MpiGroup, nelem_per_send: int, allreduce_length: int,
                           scratch_size: int, ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_allpair_allreduce("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allreduce_length={allreduce_length}, "
              f"scratch_size={scratch_size}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=ninstance,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allreduce_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allreduce_length", [128, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_ring_allreduce(mpi_group: MpiGroup, nelem_per_send: int, allreduce_length: int,
                        scratch_size: int, ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_ring_allreduce("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allreduce_length={allreduce_length}, "
              f"scratch_size={scratch_size}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
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
                              connection_types=connection_types,
                              data=memory,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allreduce_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allreduce_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_tree_allreduce(nelem_per_send: int, allreduce_length: int, scratch_size: int,
                        sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_allreduce("
              f"nelem_per_send={nelem_per_send}, "
              f"allreduce_length={allreduce_length}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    memory = cp.array([group.my_rank + 1] * allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=allreduce_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * allreduce_length, dtype=cp.int32)
    assert cp.array_equal(memory, expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("allreduce_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("iters", [2, 10])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_multrun_allreduce(nelem_per_send: int, allreduce_length: int, scratch_size: int,
                           iters: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multrun_allreduce("
              f"nelem_per_send={nelem_per_send}, "
              f"allreduce_length={allreduce_length}, "
              f"scratch_size={scratch_size}, "
              f"iters={iters}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    memory = cp.ones(allreduce_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    expected = [cp.array([group.nranks ** (i + 1)] * allreduce_length, dtype=cp.int32)
                for i in range(iters)]

    proxy_service.start_proxy()
    for i in range(iters):
        kernel(nelem_total=allreduce_length, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected[i])
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_vary_size_allreduce(nelem_per_send: int, scratch_size: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_vary_size_allreduce("
              f"nelem_per_send={nelem_per_send}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    allreduce_lengths = [3 * 2 ** n for n in range(10, 21)]
    max_length = max(allreduce_lengths)
    init_data = cp.ones(max_length, dtype=cp.int32)
    memory = cp.ones(max_length, dtype=cp.int32)

    kernel = allreduce_kernel(Ts, Cs,
                              k=3,
                              group=group,
                              connections=connections,
                              connection_types=connection_types,
                              data=memory,
                              scratch_size=scratch_size,
                              proxy_service=proxy_service)

    funcs = [kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
             for length in allreduce_lengths]
    expected = [cp.array([group.nranks] * length + [1] * (max_length - length), dtype=cp.int32)
                for i, length in enumerate(allreduce_lengths)]

    proxy_service.start_proxy()
    for i in range(len(allreduce_lengths)):
        cp.copyto(memory, init_data)
        funcs[i]()
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory, expected[i])
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("allgather_length", [128, 2 ** 20])
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_allpair_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                           ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_allpair_allgather("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
                              proxy_service=proxy_service)

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
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_ring_allgather(mpi_group: MpiGroup, nelem_per_send: int, allgather_length: int,
                        ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_ring_allgather("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
                              proxy_service=proxy_service)

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
def test_tree_allgather(nelem_per_send: int, allgather_length: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_allgather("
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
                              proxy_service=proxy_service)

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
def test_multrun_allgather(nelem_per_send: int, allgather_length: int, iters: int,
                           sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multrun_allgather("
              f"nelem_per_send={nelem_per_send}, "
              f"allgather_length={allgather_length}, "
              f"iters={iters}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
                              proxy_service=proxy_service)

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
def test_vary_size_allgather(nelem_per_send: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_vary_size_allreduce("
              f"nelem_per_send={nelem_per_send}, "
              f"sm_node_size={sm_node_size})", flush=True)
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
                              proxy_service=proxy_service)

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


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("reduce_scatter_length", [128, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_allpair_reduce_scatter(mpi_group: MpiGroup, nelem_per_send: int, reduce_scatter_length: int,
                                scratch_size: int, ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_allpair_reduce_scatter("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[(u, v)] for v in range(group.nranks) if u != v]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel(Ts, Cs,
                                   k=ninstance,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=memory,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)
    
    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
    assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@parametrize_mpi_groups(2, 8)
@pytest.mark.parametrize("nelem_per_send", [8, 20])
@pytest.mark.parametrize("reduce_scatter_length", [128, 2 ** 20])
@pytest.mark.parametrize("scratch_size", [20, 32])
@pytest.mark.parametrize("ninstance", [1, 4])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_ring_reduce_scatter(mpi_group: MpiGroup, nelem_per_send: int, reduce_scatter_length: int,
                             scratch_size: int, ninstance: int, sm_node_size: int):
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_ring_reduce_scatter("
              f"mpi_group.comm.size={mpi_group.comm.size}, "
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"ninstance={ninstance}, "
              f"sm_node_size={sm_node_size})", flush=True)
    group, connections = create_and_connect(mpi_group, "NVLink")
    connection_types = {dest: "sm" if group.my_rank // sm_node_size == dest // sm_node_size else "proxy"
                        for dest in connections.keys()}
    proxy_service = ProxyService()

    Ts = {(u, i): [[((u + d) % group.nranks, (u + d + 1) % group.nranks)]
                   for d in range(group.nranks - 1)]
          for u, i in itertools.product(range(group.nranks), range(ninstance))}
    Cs = {(u, i): 1 for u, i in itertools.product(range(group.nranks), range(ninstance))}

    memory = cp.array([group.my_rank + 1] * reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel(Ts, Cs,
                                   k=ninstance,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=memory,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)

    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
    assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("reduce_scatter_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_tree_reduce_scatter(nelem_per_send: int, reduce_scatter_length: int, scratch_size: int,
                             sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_tree_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    memory = cp.array([group.my_rank + 1] * reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel(Ts, Cs,
                                   k=3,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=memory,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)
    
    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    expected = cp.array([sum(n + 1 for n in range(group.nranks))] * shard_size, dtype=cp.int32)
    assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("reduce_scatter_length", [3 * 1024, 3 * 2 ** 20])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("iters", [2, 10])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_multrun_reduce_scatter(nelem_per_send: int, reduce_scatter_length: int, scratch_size: int,
                                iters: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_multrun_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"reduce_scatter_length={reduce_scatter_length}, "
              f"scratch_size={scratch_size}, "
              f"iters={iters}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    shard_size = reduce_scatter_length // group.nranks
    shard_begin = shard_size * group.my_rank
    shard_end = shard_begin + shard_size

    init_data = cp.ones(reduce_scatter_length, dtype=cp.int32)
    expected = cp.array([group.nranks] * shard_size, dtype=cp.int32)
    memory = cp.zeros(reduce_scatter_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel(Ts, Cs,
                                   k=3,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=memory,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)

    proxy_service.start_proxy()
    for _ in range(iters):
        cp.copyto(memory, init_data)
        kernel(nelem_total=reduce_scatter_length, nelem_per_send=nelem_per_send)
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory[shard_begin: shard_end], expected)
    proxy_service.stop_proxy()


@pytest.mark.parametrize("nelem_per_send", [256, 512])
@pytest.mark.parametrize("scratch_size", [512, 1024])
@pytest.mark.parametrize("sm_node_size", [1, 4, 8])
def test_vary_size_reduce_scatter(nelem_per_send: int, scratch_size: int, sm_node_size: int):
    mpi_group = MpiGroup(list(range(8)))
    if mpi_group.comm.rank == 0:
        print(f"TEST: test_vary_size_reduce_scatter("
              f"nelem_per_send={nelem_per_send}, "
              f"scratch_size={scratch_size}, "
              f"sm_node_size={sm_node_size})", flush=True)
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

    reduce_scatter_lengths = [3 * 2 ** n for n in range(10, 21)]
    max_length = max(reduce_scatter_lengths)
    init_data = cp.ones(max_length, dtype=cp.int32)
    memory = cp.ones(max_length, dtype=cp.int32)

    kernel = reduce_scatter_kernel(Ts, Cs,
                                   k=3,
                                   group=group,
                                   connections=connections,
                                   connection_types=connection_types,
                                   data=memory,
                                   scratch_size=scratch_size,
                                   proxy_service=proxy_service)

    funcs = [kernel.get_func(nelem_total=length, nelem_per_send=nelem_per_send)
             for length in reduce_scatter_lengths]
    shard_sizes = [length // group.nranks for length in reduce_scatter_lengths]
    expected = [cp.array([group.nranks] * shard_size, dtype=cp.int32) for shard_size in shard_sizes]
    shard_begins = [shard_size * group.my_rank for shard_size in shard_sizes]
    shard_ends = [shard_begin + shard_size for shard_begin, shard_size in zip(shard_begins, shard_sizes)]

    proxy_service.start_proxy()
    for i in range(len(reduce_scatter_lengths)):
        cp.copyto(memory, init_data)
        funcs[i]()
        cp.cuda.runtime.deviceSynchronize()  # Freeze if commented out or moved after `cp.array_equal`
        assert cp.array_equal(memory[shard_begins[i]: shard_ends[i]], expected[i])
    proxy_service.stop_proxy()
