// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <assert.h>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_reduceScatter_sendtb_schedule
#define N_PEERS 8
#endif

#define FLUSH_INTERVAL 50
#define MAX_NLOOPS 1048576

// END_DEFINES //


static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


MSCCLPP_DEVICE_INLINE void
    parallel_sm_threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, int* recv_scratch,
                                const uint64_t scratch_size, int* data, const int nrecv_peers,
                                const int sm_block_idx, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer,
                                const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (nrecv_peers == 1) {
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
      const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);

      sm_syncer->sync(sm_block_cnt);

      recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), size * sizeof(int), 
                           tid + sm_block_idx * blockDim.x, sm_block_cnt * blockDim.x);

      sm_syncer->sync(sm_block_cnt);
    }
  }
}


MSCCLPP_DEVICE_INLINE void
    sendtb_threadblockCall(mscclpp::SmChannelDeviceHandle* send_sm_channel, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, const int nrecv_proxy,
                           int** recv_scratch_arr, const uint64_t scratch_size, int* data,
                           int* pending_receives_arr, const int nrecv_sm, int* sent_progress,
                           const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send;
  assert(nloops <= MAX_NLOOPS);

  int pending_sends = 0; // only thread 0 needs this
  const int max_pending_sends = scratch_size / nelem_per_send;

  __shared__ int pending_receives_arr_local[N_PEERS];
  __shared__ int sent;
  int reduced[N_PEERS] = {};
  int sent_local = (send_sm_channel == nullptr && send_proxy_channel == nullptr ? nloops : 0);
  int min_reduced = (nrecv_sm + nrecv_proxy > 0 ? 0 : nloops);
  if (nrecv_sm == 0) {
    assert(nrecv_proxy > 0 || send_sm_channel != nullptr || send_proxy_channel != nullptr);
  } else {
    assert(nrecv_sm + nrecv_proxy > 1);
  }

  for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) pending_receives_arr_local[i] = 0;
  if (tid == 0) sent = 0;
  __syncthreads();

  while (min_reduced < nloops || sent_local < nloops) {
    if (min_reduced < nloops) {
      for (int i = tid; i < nrecv_sm; i += blockDim.x) {
        int preceived = pending_receives_arr_local[i];
        if (preceived == 0) pending_receives_arr_local[i] = *((volatile int*) &pending_receives_arr[i]);
      }
      for (int i = tid; i < nrecv_proxy; i += blockDim.x) {
        int preceived = pending_receives_arr_local[i + nrecv_sm];
        if (preceived == 0) {
          preceived += recv_proxy_channels[i].poll(nloops - preceived - reduced[i + nrecv_sm]);
          pending_receives_arr_local[i + nrecv_sm] = preceived;
        }
      }
    }
    __syncthreads();

    if (min_reduced < nloops) {
      min_reduced = nloops;
      bool reduced_round[N_PEERS];
      for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
        const int preceived = pending_receives_arr_local[i];
        assert(preceived >= 0);
        if (preceived > 0) {
          const uint64_t s_start = (reduced[i] % max_pending_sends) * nelem_per_send;
          const uint64_t d_start = data_start + reduced[i] * nelem_per_send;
          const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
          const uint64_t nElem4 = nElem / 4;
          const uint64_t nLastElem = nElem % 4;
          int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);
          int4* const scratch4 = reinterpret_cast<int4*>(&recv_scratch_arr[i][s_start]);

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

          ++reduced[i];
          reduced_round[i] = true;
          __syncthreads();
        } else {
          reduced_round[i] = false;
        }
        min_reduced = min(min_reduced, reduced[i]);
      }
      for (int i = tid; i < nrecv_sm; i += blockDim.x) {
        if (reduced_round[i]) {
          pending_receives_arr_local[i] = atomicDec((unsigned int*) &pending_receives_arr[i], 0xffffffff) - 1;
        }
      }
      for (int i = tid; i < nrecv_proxy; i += blockDim.x) {
        if (reduced_round[i + nrecv_sm]) {
          --pending_receives_arr_local[i + nrecv_sm];
          if (reduced[i + nrecv_sm] > 0 && reduced[i + nrecv_sm] % FLUSH_INTERVAL == 0) recv_proxy_channels[i].flush();
          recv_proxy_channels[i].signal();
        }
      }
    }

    if (min_reduced > sent_local) {
      bool hasSent = false;
      if (tid == 0) {
        if (send_sm_channel != nullptr) {
          send_sm_channel->signal(min_reduced - sent_local);
          sent_local = min_reduced;
          sent = sent_local;
          hasSent = true;
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
            hasSent = true;
          }
        }
        if (hasSent && (nrecv_sm + nrecv_proxy > 0)) *((volatile int*) sent_progress) = sent_local;
      }
      if (__syncthreads_or(hasSent)) sent_local = sent;
    }
  }
  if (tid == 0) {
    if (send_sm_channel != nullptr) send_sm_channel->wait();
    if (send_proxy_channel != nullptr) {
      while (pending_sends > 0) pending_sends -= send_proxy_channel->poll(pending_sends);
      send_proxy_channel->flush();
    }
  }
  for (int i = tid; i < nrecv_proxy; i += blockDim.x) recv_proxy_channels[i].flush();
}


MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int* recv_scratch, const uint64_t scratch_size, int* data,
                    int* pending_receives, const int nrecv_peers, int* sent_progress,
                    const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer, const bool skip_sm_signal,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  assert(nloops <= MAX_NLOOPS);

  int pending_sends = 0; // only thread 0 needs this
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (tid == 0) {
    assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
    assert(data_start % 4 == 0);
    assert(nelem_per_send % 4 == 0);
    assert(recv_sm_channel != nullptr);
    assert(send_sm_channel == nullptr || send_proxy_channel == nullptr);
    assert(sm_block_cnt >= 1);
    if (send_sm_channel != nullptr || send_proxy_channel != nullptr) assert(nrecv_peers == 1);
  }

  // only thread 0 needs to track folloing variables
  int ready = (skip_sm_signal ? nloops : 0);
  int preceived = 0;
  int sent = (send_sm_channel != nullptr || send_proxy_channel != nullptr ? 0 : nloops);

  __syncthreads();

  // need to track pending_receives and write to scartch only if nrecv_peers > 1
  // need to receive reduce and send only if nrecv_peers == 1
  for (int loop = 0; loop < nloops; ++loop) {
    if (tid == 0) {
      if (nrecv_peers > 1) {
        while (preceived == max_pending_sends) {
          preceived = *((volatile int*) pending_receives);
        }
      }
      while (ready == loop) ready += recv_sm_channel->poll(nloops - ready);
    }
    __syncthreads();

    const uint64_t d_start = data_start + loop * nelem_per_send;
    const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);

    if (nrecv_peers == 1) {
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

      if (tid == 0) {
        if (send_sm_channel != nullptr) {
          assert(loop == sent);
          send_sm_channel->signal();
          ++sent;
        } else if (send_proxy_channel != nullptr) {
          do {
            if (pending_sends == max_pending_sends) {
              pending_sends -= send_proxy_channel->poll(pending_sends);
            }
            if (pending_sends < max_pending_sends) {
              do {
                const uint64_t s_start = (sent % max_pending_sends) * nelem_per_send;
                const uint64_t d_start = data_start + sent * nelem_per_send;
                const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
                if (sent > 0 && sent % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
                send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
                ++pending_sends;
                ++sent;
              } while (pending_sends < max_pending_sends && sent <= loop);
            }
          } while (loop == nloops - 1 && sent <= loop); // last loop ensures all send completes
        }
      }
      __syncthreads();
    } else {
      const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;

      if (sm_block_cnt > 1) sm_syncer->sync(sm_block_cnt);
      recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), nElem * sizeof(int), tid, sm_block_cnt * blockDim.x);
      sm_syncer->sync(sm_block_cnt);
      __threadfence();

      if (tid == 0) preceived = atomicInc((unsigned int*) pending_receives, 0xffffffff) + 1;
    }
  }
  if (tid == 0) {
    if (!skip_sm_signal) recv_sm_channel->signal();
    if (send_sm_channel != nullptr) send_sm_channel->wait();
    else if (send_proxy_channel != nullptr) {
      while (pending_sends > 0) pending_sends -= send_proxy_channel->poll(pending_sends);
      send_proxy_channel->flush();
    }
  }
  __syncthreads();
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channel_indics, int* send_proxy_channel_indics,
           int* threadblock_type_block, int*** recv_scratch_arr_block, int** recv_scratch_block, const uint64_t scratch_size, int* data,
           int** pending_receives_arr_block, int** pending_receives_block,
           int* nrecv_peers_block, int* nrecv_sm_block, int* nrecv_proxy_block, int** sent_progress_block,
           int* sm_block_idx_block, int* sm_block_cnt_block, mscclpp::DeviceSyncer** sm_syncer_block, bool* skip_signal_block,
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block, const uint64_t debug_flag) {
  const int bid = blockIdx.x;
  mscclpp::SmChannelDeviceHandle* recv_sm_channel = (recv_sm_channel_indics[bid] < 0 ? nullptr : &recv_sm_channel_block[recv_sm_channel_indics[bid]]);
  mscclpp::SmChannelDeviceHandle* send_sm_channel = (send_sm_channel_indics[bid] < 0 ? nullptr : &send_sm_channel_block[send_sm_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel = (recv_proxy_channel_indics[bid] < 0 ? nullptr : &recv_proxy_channel_block[recv_proxy_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel = (send_proxy_channel_indics[bid] < 0 ? nullptr : &send_proxy_channel_block[send_proxy_channel_indics[bid]]);
  int threadblock_type = threadblock_type_block[bid];
  int** recv_scratch_arr = recv_scratch_arr_block[bid];
  int* recv_scratch = recv_scratch_block[bid];
  int* pending_receives_arr = pending_receives_arr_block[bid];
  int* pending_receives = pending_receives_block[bid];
  int nrecv_peers = nrecv_peers_block[bid];
  int nrecv_sm = nrecv_sm_block[bid];
  int nrecv_proxy = nrecv_proxy_block[bid];
  int* sent_progress = sent_progress_block[bid];
  const int sm_block_idx = sm_block_idx_block[bid];
  const int sm_block_cnt = sm_block_cnt_block[bid];
  mscclpp::DeviceSyncer* sm_syncer = sm_syncer_block[bid];
  const bool skip_signal = skip_signal_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  if (threadblock_type > 0 && nrecv_sm > 0) {
    *((volatile int*) sent_progress) = 0;
  } else if (threadblock_type == 0 && nrecv_peers > 1) {
    *((volatile int*) pending_receives) = 0;
  }
  // if (sm_block_idx == 0) new(sm_syncer) mscclpp::DeviceSyncer();
  deviceSyncer.sync(gridDim.x);

  if (threadblock_type > 0) {
    assert(recv_sm_channel == nullptr);
    assert(recv_scratch == nullptr);
    assert(pending_receives == nullptr);
    assert(nrecv_peers == nrecv_sm + nrecv_proxy);
    assert(sm_block_idx < 0);
    assert(sm_block_cnt == 0);
    assert(sm_syncer == nullptr);
    assert(!skip_signal);
    sendtb_threadblockCall(send_sm_channel, send_proxy_channel, recv_proxy_channel, nrecv_proxy,
                           recv_scratch_arr, scratch_size, data,
                           pending_receives_arr, nrecv_sm, sent_progress,
                           data_start, nelem_per_send, nelem_total, debug_flag);
  } else {
    assert(recv_proxy_channel == nullptr);
    assert(recv_scratch_arr == nullptr);
    assert(pending_receives_arr == nullptr);
    assert(nrecv_peers == nrecv_sm + nrecv_proxy);
    if (threadblock_type == 0) {
      assert(sm_block_idx == 0);
      threadblockCall(recv_sm_channel, send_sm_channel, send_proxy_channel,
                      recv_scratch, scratch_size, data,
                      pending_receives, nrecv_peers, sent_progress,
                      sm_block_cnt, sm_syncer, skip_signal,
                      data_start, nelem_per_send, nelem_total, debug_flag);
    } else {
      assert(sm_block_idx > 0);
      parallel_sm_threadblockCall(recv_sm_channel, recv_scratch, scratch_size, data, nrecv_peers,
                                  sm_block_idx, sm_block_cnt, sm_syncer,
                                  data_start, nelem_per_send, nelem_total, debug_flag);
    }
  }
}