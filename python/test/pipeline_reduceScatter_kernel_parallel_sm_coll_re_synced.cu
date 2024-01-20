// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <assert.h>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_reduceScatter_coll_re_schedule
#define N_PEERS 8
#endif

#define FLUSH_INTERVAL 50
#define MAX_NLOOPS 1048576

// #define NO_REDUCE

// END_DEFINES //


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


MSCCLPP_DEVICE_INLINE void
    parallel_sm_threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, int** recv_scratch_arr,
                                const uint64_t scratch_size, int* data, const int nrecv_peers,
                                const int reduce_block_idx, const int reduce_block_cnt, mscclpp::DeviceSyncer* reduce_syncer,
                                const int sm_block_idx, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer,
                                const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (nrecv_peers == 1 && recv_sm_channel != nullptr) {
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
  } else {
    for (int loop = 0; loop < nloops; ++loop) {
      const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
      const uint64_t d_start = data_start + loop * nelem_per_send;
      const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);

      if (recv_sm_channel != nullptr) {
        sm_syncer->sync(sm_block_cnt);
        recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), nElem * sizeof(int), 
                             tid + sm_block_idx * blockDim.x, sm_block_cnt * blockDim.x);
      }
    
      const uint64_t nElem4 = nElem / 4;
      int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);

      reduce_syncer->sync(reduce_block_cnt);

#ifndef NO_REDUCE
      for (uint64_t offset = tid + reduce_block_idx * blockDim.x; offset < nElem4; offset += reduce_block_cnt * blockDim.x) {
        int4 tmp = data4[offset];
        for (int i = 0; i < nrecv_peers; ++i) {
          int4 val = reinterpret_cast<int4*>(&recv_scratch_arr[i][s_start])[offset];
          tmp.x += val.x;
          tmp.y += val.y;
          tmp.z += val.z;
          tmp.w += val.w;
        }
        data4[offset] = tmp;
      }
#endif

      reduce_syncer->sync(reduce_block_cnt);
    }
  }
}

MSCCLPP_DEVICE_INLINE void
    parallel_reduce_threadblockCall(int** recv_scratch_arr, const uint64_t scratch_size, int* data, const int nrecv_peers,
                                    const int reduce_block_idx, const int reduce_block_cnt, mscclpp::DeviceSyncer* reduce_syncer,
                                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  const int max_pending_sends = scratch_size / nelem_per_send;
  
  for (int loop = 0; loop < nloops; ++loop) {
    const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
    const uint64_t d_start = data_start + loop * nelem_per_send;
    const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
    const uint64_t nElem4 = nElem / 4;
    int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);

    reduce_syncer->sync(reduce_block_cnt);

#ifndef NO_REDUCE
    for (uint64_t offset = tid + reduce_block_idx * blockDim.x; offset < nElem4; offset += reduce_block_cnt * blockDim.x) {
      int4 tmp = data4[offset];
      for (int i = 0; i < nrecv_peers; ++i) {
        int4 val = reinterpret_cast<int4*>(&recv_scratch_arr[i][s_start])[offset];
        tmp.x += val.x;
        tmp.y += val.y;
        tmp.z += val.z;
        tmp.w += val.w;
      }
      data4[offset] = tmp;
    }
#endif

    reduce_syncer->sync(reduce_block_cnt);
  }
}

// test_vary_size_reduce_scatter not passed!!!
MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int** recv_scratch_arr, const int nrecv_sm, const int nrecv_proxy, const int nrecv_peers,
                    const uint64_t scratch_size, int* data,
                    const int reduce_block_idx, const int reduce_block_cnt, mscclpp::DeviceSyncer* reduce_syncer,
                    const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer, const bool skip_signal,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  assert(nloops <= MAX_NLOOPS);

  int pending_sends = 0; // only thread 0 at is_first_block needs this
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (tid == 0) {
    assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
    assert(data_start % 4 == 0);
    assert(nelem_per_send % 4 == 0);
    if (reduce_block_idx != 0) {
      assert(send_sm_channel == nullptr && send_proxy_channel == nullptr);
      assert(nrecv_proxy == 0);
    } else {
      assert(send_sm_channel == nullptr || send_proxy_channel == nullptr);
      assert(nrecv_peers == nrecv_sm + nrecv_proxy);
    }
    assert((recv_sm_channel != nullptr) == (nrecv_sm > 0));
    assert((recv_proxy_channels != nullptr) == (nrecv_proxy > 0));
    assert(nrecv_peers > 0 || send_sm_channel != nullptr || send_proxy_channel != nullptr);
    if (nrecv_peers > 0) {
      assert(reduce_block_idx % sm_block_cnt == 0);
      assert(reduce_block_cnt >= max(1, nrecv_sm) * sm_block_cnt);
    } else {
      assert(reduce_block_cnt == 1);
      assert(sm_block_cnt == 1);
    }
    assert(nrecv_proxy <= blockDim.x);
    assert(nrecv_peers <= blockDim.x);
  }


  if (reduce_block_idx == 0 && nrecv_peers == 0 && send_sm_channel != nullptr) {
    if (tid == 0) send_sm_channel->signal(nloops);
  }

  for (int loop = 0; loop < nloops; ++loop) {
    const uint64_t d_start = data_start + loop * nelem_per_send;
    const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
    if (nrecv_peers > 0) {
      if (recv_sm_channel != nullptr) {
        if (!skip_signal) {
          if (tid == 0) recv_sm_channel->wait();
          __syncthreads();
        }
      }
      if (nrecv_peers == 1 && recv_sm_channel != nullptr) {
        const uint64_t d_start4 = d_start / 4;
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
        sm_syncer->sync(sm_block_cnt);
      } else {
        const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;

        if (recv_sm_channel != nullptr) {
          if (sm_block_cnt > 1) sm_syncer->sync(sm_block_cnt);
          recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), nElem * sizeof(int), tid, sm_block_cnt * blockDim.x);
        }
        
        const uint64_t nElem4 = nElem / 4;
        const uint64_t nLastElem = nElem % 4;
        int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);

        if (tid < nrecv_proxy) recv_proxy_channels[tid].wait();
        reduce_syncer->sync(reduce_block_cnt);


#ifndef NO_REDUCE
        for (uint64_t offset = tid + reduce_block_idx * blockDim.x; offset < nElem4; offset += reduce_block_cnt * blockDim.x) {
          int4 tmp = data4[offset];
          for (int i = 0; i < nrecv_peers; ++i) {
            int4 val = reinterpret_cast<int4*>(&recv_scratch_arr[i][s_start])[offset];
            tmp.x += val.x;
            tmp.y += val.y;
            tmp.z += val.z;
            tmp.w += val.w;
          }
          data4[offset] = tmp;
        }
        if (nLastElem > 0 && reduce_block_idx == 0 && tid == 0) {
          int4 tmp = data4[nElem4];
          for (int i = 0; i < nrecv_peers; ++i) {
            int4 val = reinterpret_cast<int4*>(&recv_scratch_arr[i][s_start])[nElem4];
            // assert 1 <= nLastElem <= 3
            tmp.x += val.x;
            if (nLastElem > 1) tmp.y += val.y;
            if (nLastElem > 2) tmp.z += val.z;
          }
          data4[nElem4] = tmp;
        }
#endif

        reduce_syncer->sync(reduce_block_cnt);

        if (tid < nrecv_proxy) {
          if (loop > 0 && loop % FLUSH_INTERVAL == 0) recv_proxy_channels[tid].flush();
          recv_proxy_channels[tid].signal();
        }
      }
    }
    if (reduce_block_idx == 0 && tid == 0) {
      if (send_sm_channel != nullptr) {
        send_sm_channel->signal();
      } else if (send_proxy_channel != nullptr) {
        if (pending_sends > 0 && loop % FLUSH_INTERVAL == 0) pending_sends -= send_proxy_channel->poll(pending_sends);
        while (pending_sends == max_pending_sends) {
          pending_sends -= send_proxy_channel->poll(pending_sends);
        }
        const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
        if (loop > 0 && loop % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
        send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), nElem * sizeof(int));
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (recv_sm_channel != nullptr && !skip_signal) recv_sm_channel->signal();
    if (send_sm_channel != nullptr) send_sm_channel->wait();
    if (send_proxy_channel != nullptr) {
      while (pending_sends > 0) pending_sends -= send_proxy_channel->poll(pending_sends);
      send_proxy_channel->flush();
    }
  }
  if (tid < nrecv_proxy) recv_proxy_channels[tid].flush();
}

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channels_indics, int* send_proxy_channel_indics,
           int*** recv_scratch_arr_block, int* nrecv_sm_block, int* nrecv_proxy_block, int* nrecv_peers_block,
           const uint64_t scratch_size, int* data,
           int* reduce_block_idx_block, int* reduce_block_cnt_block, mscclpp::DeviceSyncer** reduce_syncer_block, int** received_arr, int** reduced_arr,
           int* sm_block_idx_block, int* sm_block_cnt_block, mscclpp::DeviceSyncer** sm_syncer_block, int** reduce_or_get_block, bool* skip_signal_block, 
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block, const uint64_t debug_flag) {
  const int bid = blockIdx.x;
  mscclpp::SmChannelDeviceHandle* recv_sm_channel = (recv_sm_channel_indics[bid] < 0 ? nullptr : &recv_sm_channel_block[recv_sm_channel_indics[bid]]);
  int** recv_scratch_arr = recv_scratch_arr_block[bid];
  const int nrecv_sm = nrecv_sm_block[bid];
  const int nrecv_peers = nrecv_peers_block[bid];
  const int reduce_block_idx = reduce_block_idx_block[bid];
  const int reduce_block_cnt = reduce_block_cnt_block[bid];
  mscclpp::DeviceSyncer* reduce_syncer = reduce_syncer_block[bid];
  const int sm_block_idx = sm_block_idx_block[bid];
  const int sm_block_cnt = sm_block_cnt_block[bid];
  mscclpp::DeviceSyncer* sm_syncer = sm_syncer_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  if (sm_block_idx == 0) {
    mscclpp::SmChannelDeviceHandle* send_sm_channel = (send_sm_channel_indics[bid] < 0 ? nullptr : &send_sm_channel_block[send_sm_channel_indics[bid]]);
    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels = (recv_proxy_channels_indics[bid] < 0 ? nullptr : &recv_proxy_channels_block[recv_proxy_channels_indics[bid]]);
    mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel = (send_proxy_channel_indics[bid] < 0 ? nullptr : &send_proxy_channel_block[send_proxy_channel_indics[bid]]);
    const int nrecv_proxy = nrecv_proxy_block[bid];
    const bool skip_signal = skip_signal_block[bid];
    threadblockCall(recv_sm_channel, send_sm_channel, recv_proxy_channels, send_proxy_channel,
                    recv_scratch_arr, nrecv_sm, nrecv_proxy, nrecv_peers,
                    scratch_size, data,
                    reduce_block_idx, reduce_block_cnt, reduce_syncer,
                    sm_block_cnt, sm_syncer, skip_signal,
                    data_start, nelem_per_send, nelem_total, debug_flag);
  } else if (sm_block_idx > 0) {
    parallel_sm_threadblockCall(recv_sm_channel, recv_scratch_arr,
                                scratch_size, data, nrecv_peers,
                                reduce_block_idx, reduce_block_cnt, reduce_syncer,
                                sm_block_idx, sm_block_cnt, sm_syncer,
                                data_start, nelem_per_send, nelem_total, debug_flag);
  } else {
    parallel_reduce_threadblockCall(recv_scratch_arr, scratch_size, data, nrecv_peers,
                                    reduce_block_idx, reduce_block_cnt, reduce_syncer,
                                    data_start, nelem_per_send, nelem_total, debug_flag);
  }
}