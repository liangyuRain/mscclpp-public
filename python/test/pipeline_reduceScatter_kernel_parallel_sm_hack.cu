// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <assert.h>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_reduceScatter_hack_schedule
#endif

#define FLUSH_INTERVAL 50
#define MAX_NLOOPS 1048576

// END_DEFINES //


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


MSCCLPP_DEVICE_INLINE void
    parallel_sm_threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, int* data,
                                const int sm_block_idx, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer,
                                const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  for (int loop = 0; loop < nloops; ++loop) {
    const uint64_t d_start = data_start + loop * nelem_per_send;
    const uint64_t d_start4 = d_start / 4;
    const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
    const uint64_t nElem4 = nElem / 4;
    int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);

    sm_syncer->sync(sm_block_cnt);

    for (uint64_t offset = tid + sm_block_idx * blockDim.x; offset < nElem4; offset += sm_block_cnt * blockDim.x) {
      int4 tmp = data4[offset];
      int4 val = recv_sm_channel->read<int4>(d_start4 + offset);
      tmp.x += val.x;
      tmp.y += val.y;
      tmp.z += val.z;
      tmp.w += val.w;
      data4[offset] = tmp;
    }

    sm_syncer->sync(sm_block_cnt);
  }
}

// only support at most one recv_sm_channel and at most one recv_proxy_channel
MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int* proxy_recv_scratch, const bool recv_sm, const bool send_sm, const bool recv_proxy, const bool send_proxy,
                    const uint64_t scratch_size, int* data, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer, const bool skip_sm_signal,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  assert(nloops <= MAX_NLOOPS);

  const int max_pending_sends = scratch_size / nelem_per_send;

  if (tid == 0) {
    assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
    assert(data_start % 4 == 0);
    assert(nelem_per_send % 4 == 0);
    assert(!send_sm || !send_proxy);
  }

  int poll_loop_cnt = 0;

  int ready_local_sm = (recv_sm && !skip_sm_signal ? 0 : nloops);
  int ready_local_proxy = (recv_proxy ? 0 : nloops);
  if (skip_sm_signal) assert(recv_sm);

  int reduced_sm = (recv_sm ? 0 : nloops);
  int reduced_proxy = (recv_proxy ? 0 : nloops);

  int sent_local;
  if (!recv_sm && !recv_proxy && send_sm) {
    assert(!skip_sm_signal);
    if (tid == 0) send_sm_channel->signal(nloops);
    sent_local = nloops;
  } else {
    sent_local = (send_sm || send_proxy ? 0 : nloops);
  }

  __shared__ int ready_sm, ready_proxy;
  __shared__ int sent;
  __shared__ int pending_sends;
  if (tid == 0) {
    ready_sm = ready_local_sm;
    ready_proxy = ready_local_proxy;
    sent = sent_local;
    pending_sends = 0;
  }
  __syncthreads();

  while (reduced_sm < nloops || reduced_proxy < nloops || sent_local < nloops) {
    if (reduced_sm < nloops || reduced_proxy < nloops) {
      // assert recv_sm or recv_proxy
      if (recv_sm && ready_local_sm == reduced_sm) {
        if (tid == 0) {
          ready_local_sm += recv_sm_channel->poll(nloops - ready_local_sm);
          ready_sm = ready_local_sm;
        }
        __syncthreads();
        ready_local_sm = ready_sm;
      }
      if (recv_proxy && ready_local_proxy == reduced_proxy) {
        if (tid == 0) {
          ready_local_proxy += recv_proxy_channel->poll(nloops - ready_local_proxy);
          ready_proxy = ready_local_proxy;
        }
        __syncthreads();
        ready_local_proxy = ready_proxy;
      }
      const int psends = pending_sends;
      if (ready_local_sm > reduced_sm &&
          !(psends < max_pending_sends && reduced_sm > sent_local && 
            reduced_proxy == sent_local && ready_local_proxy > sent_local)) {
        const uint64_t d_start = data_start + reduced_sm * nelem_per_send;
        const uint64_t d_start4 = d_start / 4;
        const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
        const uint64_t nElem4 = nElem / 4;
        const uint64_t nLastElem = nElem % 4;
        int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);
        if (sm_block_cnt > 1) sm_syncer->sync(sm_block_cnt);
        for (uint64_t offset = tid; offset < nElem4; offset += sm_block_cnt * blockDim.x) {
          int4 tmp = data4[offset];
          int4 val = recv_sm_channel->read<int4>(d_start4 + offset);
          tmp.x += val.x;
          tmp.y += val.y;
          tmp.z += val.z;
          tmp.w += val.w;
          data4[offset] = tmp;
        }
        if (nLastElem > 0 && tid == 0) {
          int4 tmp = data4[nElem4];
          int4 val = recv_sm_channel->read<int4>(d_start4 + nElem4);
          // assert 1 <= nLastElem <= 3
          tmp.x += val.x;
          if (nLastElem > 1) tmp.y += val.y;
          if (nLastElem > 2) tmp.z += val.z;
          data4[nElem4] = tmp;
        }
        ++reduced_sm;
        if (sm_block_cnt > 1) sm_syncer->sync(sm_block_cnt);
      }
      if (ready_local_proxy > reduced_proxy &&
          !(psends < max_pending_sends && reduced_proxy > sent_local && 
            reduced_sm > sent_local)) {
        const uint64_t s_start = (reduced_proxy % max_pending_sends) * nelem_per_send;
        const uint64_t d_start = data_start + reduced_proxy * nelem_per_send;
        const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
        const uint64_t nElem4 = nElem / 4;
        const uint64_t nLastElem = nElem % 4;
        int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);
        int4* const scratch4 = reinterpret_cast<int4*>(&proxy_recv_scratch[s_start]);
        for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
          int4 tmp = data4[offset];
          int4 val = scratch4[offset];
          tmp.x += val.x;
          tmp.y += val.y;
          tmp.z += val.z;
          tmp.w += val.w;
          data4[offset] = tmp;
        }
        if (nLastElem > 0 && tid == 0) {
          int4 tmp = data4[nElem4];
          int4 val = scratch4[nElem4];
          // assert 1 <= nLastElem <= 3
          tmp.x += val.x;
          if (nLastElem > 1) tmp.y += val.y;
          if (nLastElem > 2) tmp.z += val.z;
          data4[nElem4] = tmp;
        }
        ++reduced_proxy;
        if (tid == 0) {
          if (reduced_proxy % FLUSH_INTERVAL == 0) recv_proxy_channel->flush();
          recv_proxy_channel->signal();
        }
      }
      __syncthreads();
    }

    if (send_proxy) {
      if (tid == 0) {
        const int psends = pending_sends;
        if (psends == 0) {
          poll_loop_cnt = 0;
        } else {
          ++poll_loop_cnt;
          if (poll_loop_cnt == 10) {
            pending_sends -= send_proxy_channel->poll(psends);
            poll_loop_cnt = 0;
          }
        }
      }
      __syncthreads();
    }

    const int ready_to_send = min(reduced_sm, reduced_proxy);
    if (sent_local < ready_to_send) {
      // assert send_sm or send_proxy
      if (tid == 0) {
        if (send_sm) {
          send_sm_channel->signal(ready_to_send - sent_local);
          sent_local = ready_to_send;
        } else {
          int psends = pending_sends;
          if (psends == max_pending_sends) {
            psends -= send_proxy_channel->poll(psends);
            poll_loop_cnt = 0;
          }
          if (psends < max_pending_sends) {
            do {
              const uint64_t s_start = (sent_local % max_pending_sends) * nelem_per_send;
              const uint64_t d_start = data_start + sent_local * nelem_per_send;
              const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              if (sent_local > 0 && sent_local % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
              send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
              ++psends;
              ++sent_local;
            } while (psends < max_pending_sends && sent_local < ready_to_send);
          }
          pending_sends = psends;
        }
        sent = sent_local;
      }
      __syncthreads();
      sent_local = sent;
      __syncthreads();
    }
  }
  if (tid == 0) {
    if (recv_sm && !skip_sm_signal) recv_sm_channel->signal();
    if (send_sm) send_sm_channel->wait();
    if (recv_proxy) recv_proxy_channel->flush();
    if (send_proxy) {
      int psends = pending_sends;
      while (psends > 0) psends -= send_proxy_channel->poll(psends);
      send_proxy_channel->flush();
    }
  }
  __syncthreads();
}

// __device__ mscclpp::DeviceSyncer deviceSyncer;

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channel_indics, int* send_proxy_channel_indics,
           int** proxy_recv_scratch_block, const uint64_t scratch_size, int* data,
           int* sm_block_idx_block, int* sm_block_cnt_block, mscclpp::DeviceSyncer** sm_syncer_block, bool* skip_signal_block,
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block, const uint64_t debug_flag) {
  const int bid = blockIdx.x;
  mscclpp::SmChannelDeviceHandle* recv_sm_channel = (recv_sm_channel_indics[bid] < 0 ? nullptr : &recv_sm_channel_block[recv_sm_channel_indics[bid]]);
  mscclpp::SmChannelDeviceHandle* send_sm_channel = (send_sm_channel_indics[bid] < 0 ? nullptr : &send_sm_channel_block[send_sm_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel = (recv_proxy_channel_indics[bid] < 0 ? nullptr : &recv_proxy_channel_block[recv_proxy_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel = (send_proxy_channel_indics[bid] < 0 ? nullptr : &send_proxy_channel_block[send_proxy_channel_indics[bid]]);
  int* proxy_recv_scratch = proxy_recv_scratch_block[bid];
  const bool recv_sm = (recv_sm_channel != nullptr);
  const bool send_sm = (send_sm_channel != nullptr);
  const bool recv_proxy = (recv_proxy_channel != nullptr);
  const bool send_proxy = (send_proxy_channel != nullptr);
  const int sm_block_idx = sm_block_idx_block[bid];
  const int sm_block_cnt = sm_block_cnt_block[bid];
  mscclpp::DeviceSyncer* sm_syncer = sm_syncer_block[bid];
  const bool skip_signal = skip_signal_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  // Assume sm_syncer has been initialized in python code by zeroing all bytes
  // if (sm_block_idx == 0 && sm_block_cnt > 1) new(sm_syncer) mscclpp::DeviceSyncer();
  // deviceSyncer.sync(gridDim.x);

  if (sm_block_idx == 0) {
    threadblockCall(recv_sm_channel, send_sm_channel, recv_proxy_channel, send_proxy_channel,
                    proxy_recv_scratch, recv_sm, send_sm, recv_proxy, send_proxy,
                    scratch_size, data, sm_block_cnt, sm_syncer, skip_signal,
                    data_start, nelem_per_send, nelem_total, debug_flag);
  } else {
    parallel_sm_threadblockCall(recv_sm_channel, data,
                                sm_block_idx, sm_block_cnt, sm_syncer,
                                data_start, nelem_per_send, nelem_total, debug_flag);
  }
}