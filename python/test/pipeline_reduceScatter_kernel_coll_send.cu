// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <assert.h>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_reduceScatter_schedule
#endif

#define FLUSH_INTERVAL 50
#define MAX_NLOOPS 1048576

// locked: *send_status == (SEND_LOCK_MASK | ((uint64_t) sent_local)
// unlocked: *send_status == ((uint64_t) pending_sends << 32) | ((uint64_t) sent_local)
#define SEND_LOCK_MASK 0xffffffff00000000

// END_DEFINES //


MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int* recv_scratch, const bool recv_sm, const bool send_sm, const bool recv_proxy, const bool send_proxy,
                    const uint64_t scratch_size, int* data,
                    int* reduce_counts, const int nrecv_peers, uint64_t* const send_status, const bool is_first_block,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t debug_flag) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  assert(nloops <= MAX_NLOOPS);

  const int max_pending_sends = scratch_size / nelem_per_send;

  if (tid == 0) {
    assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
    assert(data_start % 4 == 0);
    assert(nelem_per_send % 4 == 0);
    assert((recv_sm || recv_proxy) == (nrecv_peers > 0));
    assert(!send_sm || !send_proxy);
    assert(!recv_sm || !recv_proxy);
  }

  int received = (recv_sm || recv_proxy ? 0 : nloops);
  int reduced = (recv_sm || recv_proxy ? 0 : nloops);

  int sent_local;
  if (!recv_sm && !recv_proxy && send_sm) {
    if (tid == 0) send_sm_channel->signal(nloops);
    sent_local = nloops;
  } else {
    sent_local = (send_sm || send_proxy ? 0 : nloops);
  }

  __shared__ int ready;
  __shared__ int sent;
  if (tid == 0) {
    ready = (recv_sm || recv_proxy ? 0 : nloops);
    sent = sent_local;
  }

  int pending_sends = 0; // local copy of pending_sends
  __syncthreads();

  while (reduced < nloops || sent_local < nloops) {
    if (received < nloops) {
      // assert recv_sm or recv_proxy
      int ready_local = ready;
      if (ready_local == received) {
        if (tid == 0) {
          if (recv_sm) {
            ready_local += recv_sm_channel->poll(nloops - ready);
          } else {
            ready_local += recv_proxy_channel->poll(nloops - ready);
          }
          ready = ready_local;
        }
        __syncthreads();
        ready_local = ready;
      }
      if (ready_local > received) {
        if (recv_sm && nrecv_peers > 1) {
          // only recv sm channel with no rrcs use get
          // recv sm channel with rrcs will directly use read when reduce
          // proxy channel does not have get
          const uint64_t s_start = (received % max_pending_sends) * nelem_per_send;
          const uint64_t d_start = data_start + received * nelem_per_send;
          const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
          recv_sm_channel->get(d_start * sizeof(int), s_start * sizeof(int), size * sizeof(int), tid, blockDim.x);
          ++received;
        } else {
          received = ready_local;
        }
      }
      __syncthreads();
    }

    if (reduced < received) {
      const uint64_t s_start = (reduced % max_pending_sends) * nelem_per_send;
      const uint64_t d_start = data_start + reduced * nelem_per_send;
      const uint64_t d_start4 = d_start / 4;
      const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
      const uint64_t nElem4 = nElem / 4;
      const uint64_t nLastElem = nElem % 4;
      int4* const data4 = reinterpret_cast<int4*>(&data[d_start]);
      if (recv_sm && nrecv_peers == 1) {
        // no __threadfence() needed, only one threadblock is writing
        for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
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
        ++reduced;
        __syncthreads();
        if (tid == 0 && send_sm) send_sm_channel->signal();
      } else {
        // Try lock
        __shared__ int re_cnt;
        if (tid == 0) re_cnt = atomicExch(&reduce_counts[reduced], -1); // Try lock
        __syncthreads();
        const int re_cnt_local = re_cnt;
        if (re_cnt_local >= 0) {
          int4* const scratch4 = reinterpret_cast<int4*>(&recv_scratch[s_start]);
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

          // Unlock
          __syncthreads();
          if (tid == 0) {
            if (re_cnt_local + 1 == nrecv_peers && send_sm) send_sm_channel->signal();
            __threadfence();
            *((volatile int*) &reduce_counts[reduced]) = re_cnt_local + 1;
            if (recv_proxy) {
              if (reduced > 0 && reduced % FLUSH_INTERVAL == 0) recv_proxy_channel->flush();
              recv_proxy_channel->signal();
            }
          }
          ++reduced;
        }
      }
      // sm send only needs to signal and no need to track pending_sends
      // which every threadblock reduce the final recv peer will directly signal
      // other threadblocks can assume the data is send
      if (send_sm) sent_local = reduced;
      __syncthreads();
    }

    if (send_proxy) {
      if (tid == 0) {
        if (sent_local < reduced) {
          if (nrecv_peers <= 1) {
            if (pending_sends == max_pending_sends) {
              pending_sends -= send_proxy_channel->poll(pending_sends);
            }
            while (pending_sends < max_pending_sends && sent_local < reduced) {
              const uint64_t s_start = (sent_local % max_pending_sends) * nelem_per_send;
              const uint64_t d_start = data_start + sent_local * nelem_per_send;
              const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              if (sent_local > 0 && sent_local % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
              send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
              ++pending_sends;
              ++sent_local;
            }
          } else {
            uint64_t send_status_local = *((volatile uint64_t*) send_status);
            sent_local = (int) (send_status_local & 0xffffffff);
            if ((send_status_local & SEND_LOCK_MASK) != SEND_LOCK_MASK &&
                sent_local < reduced && *((volatile int*) &reduce_counts[sent_local]) == nrecv_peers) {
              send_status_local = atomicOr((unsigned long long int*) send_status, SEND_LOCK_MASK);
              if ((send_status_local & SEND_LOCK_MASK) != SEND_LOCK_MASK) { // lock success
                sent_local = (int) (send_status_local & 0xffffffff);
                pending_sends = (int) (send_status_local >> 32);
                if (pending_sends == max_pending_sends) {
                  pending_sends -= send_proxy_channel->poll(pending_sends);
                }
                while (pending_sends < max_pending_sends && sent_local < reduced && *((volatile int*) &reduce_counts[sent_local]) == nrecv_peers) {
                  const uint64_t s_start = (sent_local % max_pending_sends) * nelem_per_send;
                  const uint64_t d_start = data_start + sent_local * nelem_per_send;
                  const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
                  if (sent_local > 0 && sent_local % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
                  send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
                  ++sent_local;
                  ++pending_sends;
                }
                send_status_local = ((uint64_t) pending_sends << 32) | ((uint64_t) sent_local);
                __threadfence();
                *((volatile uint64_t*) send_status) = send_status_local; // release lock
              }
            }
          }
          sent = sent_local;
        }
      }
      __syncthreads();
      sent_local = sent;
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (recv_sm) recv_sm_channel->signal();
    if (send_sm && is_first_block) send_sm_channel->wait();
    if (recv_proxy) recv_proxy_channel->flush();
    if (send_proxy) {
      // all threadblocks have finished send
      const uint64_t send_status_local = atomicOr((unsigned long long int*) send_status, SEND_LOCK_MASK);
      if ((send_status_local & SEND_LOCK_MASK) != SEND_LOCK_MASK) {
        pending_sends = (int) (send_status_local >> 32);
        while (pending_sends > 0) pending_sends -= send_proxy_channel->poll(pending_sends);
        send_proxy_channel->flush();
      }
    }
  }
  __syncthreads();
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

MSCCLPP_DEVICE_INLINE void zero_memory(int* data, const uint64_t nelem) {
  const int tid = threadIdx.x;
  int4* data4 = reinterpret_cast<int4*>(data);
  const uint64_t nElem4 = nelem / 4;
  const uint64_t nLastElem = nelem % 4;
  for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
    int4 tmp = data4[offset];
    tmp.x = 0;
    tmp.y = 0;
    tmp.z = 0;
    tmp.w = 0;
    data4[offset] = tmp;
  }
  if (nLastElem > 0 && tid == 0) {
    int4 tmp = data4[nElem4];
    tmp.x = 0;
    if (nLastElem > 1) tmp.y = 0;
    if (nLastElem > 2) tmp.z = 0;
    data4[nElem4] = tmp;
  }
}

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channel_indics, int* send_proxy_channel_indics,
           int** recv_scratch_block, const uint64_t scratch_size, int* data,
           int** reduce_counts_block, int* nrecv_peers_block, uint64_t** send_status_block, bool* first_block,
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block, const uint64_t debug_flag) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const bool is_first_block = first_block[bid];
  mscclpp::SmChannelDeviceHandle* recv_sm_channel = (recv_sm_channel_indics[bid] < 0 ? nullptr : &recv_sm_channel_block[recv_sm_channel_indics[bid]]);
  mscclpp::SmChannelDeviceHandle* send_sm_channel = (send_sm_channel_indics[bid] < 0 ? nullptr : &send_sm_channel_block[send_sm_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel = (recv_proxy_channel_indics[bid] < 0 ? nullptr : &recv_proxy_channel_block[recv_proxy_channel_indics[bid]]);
  mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel = (send_proxy_channel_indics[bid] < 0 ? nullptr : &send_proxy_channel_block[send_proxy_channel_indics[bid]]);
  int* recv_scratch = recv_scratch_block[bid];
  const bool recv_sm = (recv_sm_channel != nullptr);
  const bool send_sm = (send_sm_channel != nullptr);
  const bool recv_proxy = (recv_proxy_channel != nullptr);
  const bool send_proxy = (send_proxy_channel != nullptr);
  int* reduce_counts = reduce_counts_block[bid];
  const int nrecv_peers = nrecv_peers_block[bid];
  uint64_t* send_status = send_status_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  if (is_first_block) {
    if (recv_sm || recv_proxy) {
      const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send;
      zero_memory(reduce_counts, nloops);
    }
    if (tid == 0 && send_proxy) *send_status = 0;
  }
  deviceSyncer.sync(gridDim.x);


  threadblockCall(recv_sm_channel, send_sm_channel, recv_proxy_channel, send_proxy_channel,
                  recv_scratch, recv_sm, send_sm, recv_proxy, send_proxy,
                  scratch_size, data,
                  reduce_counts, nrecv_peers, send_status, is_first_block,
                  data_start, nelem_per_send, nelem_total, debug_flag);
}