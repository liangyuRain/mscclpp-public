#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL allgather_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* sm_channels, mscclpp::DeviceSyncer* syncers,
           const uint64_t n_parallel_sm_blocks, const uint64_t local_offset,
           const uint64_t* offsets, const uint64_t nelem_per_channel) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int peer = bid / n_parallel_sm_blocks; // assert len(sm_channels) % n_parallel_sm_blocks == 0
    const int peer_block_idx = bid % n_parallel_sm_blocks;

    // if (peer_block_idx == 0 && tid == 0) {
    //     sm_channels[peer].signal();
    //     sm_channels[peer].wait();
    // }

    sm_channels[peer].put(local_offset * sizeof(int), nelem_per_channel * sizeof(int),
                          tid + peer_block_idx * blockDim.x, n_parallel_sm_blocks * blockDim.x);

    if (peer_block_idx == 0 && tid == 0) {
        sm_channels[peer].signal();
        sm_channels[peer].wait();
    }
}