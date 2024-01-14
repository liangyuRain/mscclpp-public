import cupy as cp
import pytest
import itertools

from mscclpp import ProxyService
from .mscclpp_mpi import MpiGroup, parametrize_mpi_groups, mpi_group
from .test_mscclpp import create_and_connect

from .pipeline_schedule import ReduceScatterPipelineKernel


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
    ntrees = 1

    kernel = ReduceScatterPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                         memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                                         recv_sm_scratches, recv_proxy_scratches, ntrees)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.array_equal(memory, cp.array([group.my_rank + 1] * nelem_total, dtype=cp.int32))
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
    ntrees = 1

    kernel = ReduceScatterPipelineKernel(recv_sm_channels, send_sm_channels, recv_proxy_channels, send_proxy_channels,
                                         memory, data_chunk_offsets, data_chunk_sizes, total_chunks, scratch_size,
                                         recv_sm_scratches, recv_proxy_scratches, ntrees)
    
    proxy_service.start_proxy()
    group.barrier()
    kernel(nelem_total=nelem_total, nelem_per_send=nelem_per_send)
    cp.cuda.runtime.deviceSynchronize()
    if group.my_rank == 0:
        assert cp.array_equal(memory, cp.array([group.nranks] * nelem_total, dtype=cp.int32))
    else:
        assert cp.array_equal(memory, cp.ones(nelem_total, dtype=cp.int32))
    proxy_service.stop_proxy()
