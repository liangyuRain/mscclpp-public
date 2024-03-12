#include <cuda_fp16.h>
#include <assert.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL reduceScatter_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");

__device__ mscclpp::DeviceSyncer syncer;


MSCCLPP_DEVICE_INLINE void reduce(mscclpp::SmChannelDeviceHandle* sm_channel,
                                  const int n_parallel_sm_blocks, const int parallel_block_idx,
                                  const uint64_t begin_offset, const uint64_t nelem,
                                  __half* data) {
    const int tid = threadIdx.x;
    float4* const data4 = reinterpret_cast<float4*>(&data[begin_offset]);
    for (uint64_t offset = tid + parallel_block_idx * blockDim.x; 
         offset < nelem / 8; 
         offset += n_parallel_sm_blocks * blockDim.x) {
        float4 tmp = data4[offset];
        float4 val = sm_channel->read<float4>(begin_offset / 8 + offset);
        *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
        *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
        *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
        *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        data4[offset] = tmp;
    }
}


// Recursive Halving
extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* sm_channels,
           const uint64_t rank, const uint64_t nranks,
           __half* data, const uint64_t nelem_per_shard) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n_parallel_sm_blocks = gridDim.x;
    const int parallel_block_idx = bid;

    uint64_t begin_offset = 0;
    uint64_t data_size = nelem_per_shard * nranks;
    uint64_t group_size = nranks;

    while (group_size > 1) {
        group_size /= 2;
        data_size /= 2;
        mscclpp::SmChannelDeviceHandle* sm_channel;
        if ((rank / group_size) % 2 == 0) {
            sm_channel = &sm_channels[rank + group_size - 1];
        } else {
            begin_offset += data_size;
            sm_channel = &sm_channels[rank - group_size];
        }
        if (parallel_block_idx == 0 && tid == 0) {
            sm_channel->signal();
            sm_channel->wait();
        }
        syncer.sync(n_parallel_sm_blocks);
        reduce(sm_channel, n_parallel_sm_blocks, parallel_block_idx,
               begin_offset, data_size, data);
        syncer.sync(n_parallel_sm_blocks);
    }

    if (parallel_block_idx == 0 && tid < nranks - 1) {
        sm_channels[tid].signal();
        sm_channels[tid].wait();
    }

    syncer.sync(n_parallel_sm_blocks);
}