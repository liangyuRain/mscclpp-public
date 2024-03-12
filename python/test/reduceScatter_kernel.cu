#include <cuda_fp16.h>
#include <assert.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL reduceScatter_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");

__device__ mscclpp::DeviceSyncer syncer;

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* sm_channels,
           const uint64_t rank, const uint64_t nranks,
           __half* data, const uint64_t nelem_per_shard) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int n_parallel_sm_blocks = gridDim.x;
    const int parallel_block_idx = bid;

    const uint64_t local_offset = rank * nelem_per_shard;

    if (parallel_block_idx == 0 && tid < nranks - 1) {
        sm_channels[tid].signal();
        sm_channels[tid].wait();
    }

    syncer.sync(n_parallel_sm_blocks);

    float4* const data4 = reinterpret_cast<float4*>(&data[local_offset]);

    assert(local_offset % 8 == 0 && nelem_per_shard % 8 == 0);
    const uint64_t local_offset4 = local_offset / 8;
    const uint64_t nelem_per_shard4 = nelem_per_shard / 8;
    for (uint64_t offset = tid + parallel_block_idx * blockDim.x; 
         offset < nelem_per_shard4; 
         offset += n_parallel_sm_blocks * blockDim.x) {
        float4 tmp = data4[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            float4 val = sm_channels[(rank + i) % (nranks - 1)].read<float4>(local_offset4 + offset);
            *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
            *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
            *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
            *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        }
        data4[offset] = tmp;
    }

    syncer.sync(n_parallel_sm_blocks);

    if (parallel_block_idx == 0 && tid < nranks - 1) {
        sm_channels[tid].signal();
        sm_channels[tid].wait();
    }

    syncer.sync(n_parallel_sm_blocks);
}