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

// END_DEFINES //


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


MSCCLPP_DEVICE_INLINE void
    parallel_sm_threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, int** recv_scratch_arr,
                                const uint64_t scratch_size, int* data, const int nrecv_sm, const int nrecv_peers,
                                const int reduce_block_idx, const int reduced_block_cnt, int* received_arr,
                                const int sm_block_idx, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer, int* reduce_or_get,
                                const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (nrecv_sm == 1) {
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
    int received = 0, reduced = 0;
    while (reduced < nloops) {
      const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
      const uint64_t d_start = data_start + loop * nelem_per_send;
      const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);

      sm_syncer->sync(sm_block_cnt);

      if (*((volatile int*) reduce_or_get) < 0) {
        const uint64_t s_start = (received % max_pending_sends) * nelem_per_send;
        const uint64_t d_start = data_start + received * nelem_per_send;
        const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
        recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), size * sizeof(int), 
                             tid + sm_block_idx * blockDim.x, sm_block_cnt * blockDim.x);
        ++received;
      } else {
        const uint64_t s_start = (reduced % max_pending_sends) * nelem_per_send;
        const uint64_t d_start = data_start + reduced * nelem_per_send;
        const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
        const uint64_t nElem4 = nElem / 4;
        int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);

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

        ++reduced;
      }

      sm_syncer->sync(sm_block_cnt);
    }
  }
}


MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int** recv_scratch_arr, const int nrecv_sm, const int nrecv_proxy, const int nrecv_peers,
                    const uint64_t scratch_size, int* data,
                    const int reduce_block_idx, const int reduced_block_cnt, int* received_arr, int* reduced_arr,
                    const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer, int* reduce_or_get, const bool skip_signal,
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
    if (nrecv_sm == 1) {
      assert(received_arr == nullptr);
      assert(reduced_arr == nullptr);
    }
    assert(reduced_block_cnt == nrecv_sm * sm_block_cnt);
    assert(reduce_block_idx % sm_block_cnt == 0);
    assert(nrecv_proxy <= blockDim.x);
    assert(nrecv_peers <= blockDim.x);
  }
  
  const int sm_peer_idx = reduce_block_idx / sm_block_cnt;

  int received_proxy = (nrecv_proxy > 0 && tid < nrecv_proxy ? 0 : nloops); // each thread track at most one recv proxy

  int received_sm = (recv_sm_channel != nullptr && !skip_signal ? 0 : nloops);
  int reduced = (recv_sm_channel != nullptr || recv_proxy_channel != nullptr ? 0 : nloops);

  int sent_local;
  if (reduce_block_idx == 0) {
    if (nrecv_peers == 0) {
      assert(!skip_signal);
      if (tid == 0) send_sm_channel->signal(nloops);
      sent_local = nloops;
    } else {
      sent_local = (send_sm_channel != nullptr || send_proxy_channel != nullptr ? 0 : nloops);
    }
  } else {
    sent_local = nloops;
  }

  __shared__ int sent;
  if (tid == 0) sent = sent_local;
  __syncthreads();

  while (reduced < nloops || sent_local < nloops) {
    if (reduce_block_idx == 0) {
      bool has_update = false;
      if (tid < nrecv_proxy) {
        if (received_proxy < nloops) {
          const int update = recv_proxy_channels[tid]->poll(nloops - received_proxy);
          if (update > 0) {
            received_proxy += update;
            has_update = true;
          }
        }
      }
      if (__syncthreads_or(has_update) && tid < nrecv_proxy) {
        *((volatile int*) &received_arr[nrecv_sm + tid]) = received_proxy;
      }
    }

    if (received_sm < nloops) {
      bool has_update = false;
      if (tid == 0) {
        if (recv_sm_channel->poll()) has_update = true;
      }
      if (__syncthreads_or(has_update)) {
        if (nrecv_sm == 1) {
          const uint64_t d_start = data_start + reduced * nelem_per_send;
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
          ++received_sm;
          ++reduced;
          sm_syncer->sync(sm_block_cnt);
        } else {
          const uint64_t s_start = (received_sm % max_pending_sends) * nelem_per_send;
          const uint64_t d_start = data_start + received_sm * nelem_per_send;
          const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);

          if (sm_block_cnt > 1) {
            *((volatile int*) reduce_or_get) = -1;
            sm_syncer->sync(sm_block_cnt);
          }
          recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), size * sizeof(int), tid, sm_block_cnt * blockDim.x);
          sm_syncer->sync(sm_block_cnt);

          ++received_sm;
          if (tid == 0) *((volatile int*) &received_arr[sm_peer_idx]) = received_sm;
        }
      }
    }

    if (nrecv_peers > 1 && __syncthreads_and(received_proxy > reduced && received_sm > reduced)) {
      const int received = (tid < nrecv_peers ? *((volatile int*) &received_arr[tid]) : nloops);

      if (sm_block_cnt > 1) *((volatile int*) reduce_or_get) = 1;
      while (__syncthreads_and(received > reduced)) {
        if (sm_block_cnt > 1) sm_syncer->sync(sm_block_cnt);

        const uint64_t s_start = (reduced % max_pending_sends) * nelem_per_send;
        const uint64_t d_start = data_start + reduced * nelem_per_send;
        const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
        const uint64_t nElem4 = nElem / 4;
        int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);
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
            int4 val = reinterpret_cast<int4*>(&recv_scratch_arr[i][nElem4])[offset];
            // assert 1 <= nLastElem <= 3
            tmp.x += val.x;
            if (nLastElem > 1) tmp.y += val.y;
            if (nLastElem > 2) tmp.z += val.z;
          }
          data4[nElem4] = tmp;
        }

        ++reduced;
        sm_syncer->sync(sm_block_cnt);

        if (tid == 0) *((volatile int*) &reduced_arr[sm_peer_idx]) = reduced;
        if (send_sm_channel != nullptr || send_proxy_channel != nullptr) break;
      }
    }

    if (reduced > sent_local) {
      int min_reduced = reduced;
      if (nrecv_sm > 1) {
        __shared__ int reduced_arr_shared[N_PEERS];
        int reduced_local = nloops;
        if (tid < nrecv_sm) {
          reduced_arr_shared[tid] = reduced_local = *((volatile int*) &reduced_arr[tid]);
        }
        if (__syncthreads_or(reduced_local == sent_local)) {
          min_reduced = sent_local;
        } else {
          for (int i = 0; i < nrecv_sm; ++i) {
            min_reduced = min(min_reduced, reduced_arr_shared[i]);
          }
        }
      }
      if (min_reduced > 0) {
        if (tid == 0) {
          if (send_sm_channel != nullptr) {
            send_sm_channel->signal(min_reduced - sent_local);
            sent_local = min_reduced;
            sent = sent_local;
          } else {
            if (pending_sends == max_pending_sends) {
              pending_sends -= send_proxy_channel->poll(pending_sends);
            }
            if (pending_sends < max_pending_sends) {
              do {
                const uint64_t s_start = (sent_local % max_pending_sends) * nelem_per_send;
                const uint64_t d_start = data_start + sent_local * nelem_per_send;
                const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
                if (sent_local > 0 && sent_local % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
                send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
                ++pending_sends;
                ++sent_local;
              } while (pending_sends < max_pending_sends && sent_local < min_reduced);
              sent = sent_local;
            }
          }
        }
        __syncthreads();
        sent_local = sent;
      }
    }
    __syncthreads();
  }
  if (reduce_block_idx == 0 && nrecv_sm > 1) {
    int reduced_local;
    do {
      reduced_local = (tid < nrecv_sm ? *((volatile int*) &reduced_arr[tid]) : nloops);
    } while(__syncthreads_or(reduced_local < nloops));
    if (tid < nrecv_peers) *((volatile int*) &received_arr[tid]) = 0;
    if (tid < nrecv_sm) *((volatile int*) &reduced_arr[tid]) = 0;
  }
  if (tid == 0) {
    if (recv_sm_channel != nullptr && !skip_signal) recv_sm_channel->signal();
    if (send_sm_channel != nullptr) send_sm_channel->wait();
    for (int i = tid; i < nrecv_proxy; i += blockDim.x) recv_proxy_channels[i]->wait();
    if (send_proxy_channel != nullptr) {
      while (pending_sends > 0) pending_sends -= send_proxy_channel->poll(pending_sends);
      send_proxy_channel->flush();
    }
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channels_indics, int* send_proxy_channel_indics,
           int*** recv_scratch_arr_block, int* nrecv_sm_block, int* nrecv_proxy_block, int* nrecv_peers_block,
           const uint64_t scratch_size, int* data,
           int* reduce_block_idx_block, int* reduced_block_cnt_block, int** received_arr, int** reduced_arr,
           int* sm_block_idx_block, int* sm_block_cnt_block, mscclpp::DeviceSyncer** sm_syncer_block, int** reduce_or_get, bool* skip_signal_block, 
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block, const uint64_t debug_flag) {
  const int bid = blockIdx.x;
  mscclpp::SmChannelDeviceHandle* recv_sm_channel = (recv_sm_channel_indics[bid] < 0 ? nullptr : &recv_sm_channel_block[recv_sm_channel_indics[bid]]);
  mscclpp::SmChannelDeviceHandle* send_sm_channel = (send_sm_channel_indics[bid] < 0 ? nullptr : &send_sm_channel_block[send_sm_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels = (recv_proxy_channels_indics[bid] < 0 ? nullptr : &recv_proxy_channels_block[recv_proxy_channels_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel = (send_proxy_channel_indics[bid] < 0 ? nullptr : &send_proxy_channel_block[send_proxy_channel_indics[bid]]);
  int** recv_scratch_arr = recv_scratch_arr_block[bid];
  const int nrecv_sm = nrecv_sm_block[bid];
  const int nrecv_proxy = nrecv_proxy_block[bid];
  const int nrecv_peers = nrecv_peers_block[bid];
  const int reduce_block_idx = reduce_block_idx_block[bid];
  const int reduced_block_cnt = reduced_block_cnt_block[bid];
  int* received = received_arr[bid];
  int* reduced = reduced_arr[bid];
  const int sm_block_idx = sm_block_idx_block[bid];
  const int sm_block_cnt = sm_block_cnt_block[bid];
  mscclpp::DeviceSyncer* sm_syncer = sm_syncer_block[bid];
  int* reduce_or_get_ptr = reduce_or_get[bid];
  const bool skip_signal = skip_signal_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  if (sm_block_idx == 0) {
    threadblockCall(recv_sm_channel, send_sm_channel, recv_proxy_channels, send_proxy_channel,
                    recv_scratch_arr, nrecv_sm, nrecv_proxy, nrecv_peers,
                    scratch_size, data,
                    reduce_block_idx, reduced_block_cnt, received, reduced,
                    sm_block_cnt, sm_syncer, reduce_or_get_ptr, skip_signal,
                    data_start, nelem_per_send, nelem_total, debug_flag);
  } else {
    parallel_sm_threadblockCall(recv_sm_channel, recv_scratch_arr,
                                scratch_size, data, nrecv_sm, nrecv_peers,
                                sm_block_idx, sm_block_cnt, sm_syncer,
                                data_start, nelem_per_send, nelem_total, debug_flag);
  }
}