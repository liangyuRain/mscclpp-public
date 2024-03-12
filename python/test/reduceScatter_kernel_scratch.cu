#include <cuda_fp16.h>
#include <assert.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL reduceScatter_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");

__device__ mscclpp::DeviceSyncer global_syncer;

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* send_sm_channels,
           mscclpp::SmChannelDeviceHandle* recv_sm_channels,
           mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels,
           const uint64_t rank, const uint64_t nranks,
           mscclpp::DeviceSyncer* syncers, __half* data, __half** scratches,
           const uint64_t n_parallel_sm_blocks, const uint64_t nelem_per_shard, const uint64_t async) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (async) {
        if (bid == 0 && tid < nranks - 1) {
            send_proxy_channels[tid].signal();
            recv_proxy_channels[tid].wait();
            const int remoteRank = tid < rank ? tid : tid + 1;
            send_proxy_channels[tid].putWithSignal(0, remoteRank * nelem_per_shard * sizeof(__half), nelem_per_shard * sizeof(__half));
            recv_proxy_channels[tid].wait();
        }
    } else {
        const int peer = bid / n_parallel_sm_blocks; // assert len(sm_channels) % n_parallel_sm_blocks == 0
        const int remoteRank = peer < rank ? peer : peer + 1;
        const int peer_block_idx = bid % n_parallel_sm_blocks;
        
        if (peer_block_idx == 0 && tid == 0) {
            send_sm_channels[peer].signal();
            recv_sm_channels[peer].wait();
        }

        syncers[peer].sync(n_parallel_sm_blocks);

        send_sm_channels[peer].put(0, remoteRank * nelem_per_shard * sizeof(__half), nelem_per_shard * sizeof(__half),
                                   tid + peer_block_idx * blockDim.x, n_parallel_sm_blocks * blockDim.x);
        
        syncers[peer].sync(n_parallel_sm_blocks);

        if (peer_block_idx == 0 && tid == 0) {
            send_sm_channels[peer].signal();
            recv_sm_channels[peer].wait();
        }
    }

    global_syncer.sync(gridDim.x);

    float4* const data4 = reinterpret_cast<float4*>(&data[rank * nelem_per_shard]);
    for (uint64_t offset = tid + bid * blockDim.x; offset < nelem_per_shard / 8; offset += gridDim.x * blockDim.x) {
        float4 tmp = data4[offset];
        for (int i = 0; i < nranks - 1; ++i) {
            float4 val = ((float4*) scratches[i])[offset];
            *reinterpret_cast<__half2*>(&tmp.x) += *reinterpret_cast<__half2*>(&val.x);
            *reinterpret_cast<__half2*>(&tmp.y) += *reinterpret_cast<__half2*>(&val.y);
            *reinterpret_cast<__half2*>(&tmp.z) += *reinterpret_cast<__half2*>(&val.z);
            *reinterpret_cast<__half2*>(&tmp.w) += *reinterpret_cast<__half2*>(&val.w);
        }
        data4[offset] = tmp;
    }

    global_syncer.sync(gridDim.x);
}