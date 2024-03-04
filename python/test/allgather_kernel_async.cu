#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

#ifndef PARAMETRIZE
#define KERNEL allgather_kernel
#endif


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SimpleProxyChannelDeviceHandle* proxy_channels, const uint64_t nchannels,
           const uint64_t local_offset, const uint64_t nelem_per_channel) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid == 0 && tid < nchannels) {
        // proxy_channels[tid].signal();
        // proxy_channels[tid].wait();
        proxy_channels[tid].putWithSignal(local_offset * sizeof(int), nelem_per_channel * sizeof(int));
        proxy_channels[tid].wait();
    }
}