#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL allgather_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


__device__ uint64_t globalFlag = 1;


extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* sm_channels, mscclpp::DeviceSyncer* syncers,
           const uint64_t n_parallel_sm_blocks, const uint64_t* offsets, const uint64_t local_offset,
           const uint64_t nelem_per_channel, const uint64_t nelem_total) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int peer = bid / n_parallel_sm_blocks; // assert len(sm_channels) % n_parallel_sm_blocks == 0
    const int peer_block_idx = bid % n_parallel_sm_blocks;
    const int flag = (int) globalFlag;
    syncers[0].sync(gridDim.x);

    // Each uint32_t has one additional flag uint32_t to indicate if the uint32_t has been written
    sm_channels[peer].putPackets(2 * local_offset * sizeof(int), local_offset * sizeof(int),
                                 nelem_per_channel * sizeof(int), tid + peer_block_idx * blockDim.x,
                                 n_parallel_sm_blocks * blockDim.x, flag);
    sm_channels[peer].getPackets(2 * offsets[peer] * sizeof(int), offsets[peer] * sizeof(int),
                                 nelem_per_channel * sizeof(int), tid + peer_block_idx * blockDim.x,
                                 n_parallel_sm_blocks * blockDim.x, flag);
    // sm_channels[peer].getPackets(0, 0, local_offset * sizeof(int), tid + bid * blockDim.x,
    //                              gridDim.x * blockDim.x, flag);
    // sm_channels[peer].getPackets(2 * (local_offset + nelem_per_channel) * sizeof(int),
    //                              (local_offset + nelem_per_channel) * sizeof(int),
    //                              (nelem_total - local_offset - nelem_per_channel) * sizeof(int),
    //                              tid + bid * blockDim.x, gridDim.x * blockDim.x, flag);

    if (tid == 0 && bid == 0) ++globalFlag;
    syncers[0].sync(gridDim.x);
}