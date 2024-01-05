// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/proxy_channel_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1)
    proxy_bug(mscclpp::SimpleProxyChannelDeviceHandle* channels) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  if (tid == 0) {
    channels[bid].signal();
    channels[bid].wait();
  }
}
