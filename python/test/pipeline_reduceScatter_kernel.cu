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

// END_DEFINES //


MSCCLPP_DEVICE_INLINE void
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel, mscclpp::SmChannelDeviceHandle* send_sm_channel,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel,
                    int* recv_scratch, const bool recv_sm, const bool send_sm, const bool recv_proxy, const bool send_proxy,
                    const uint64_t scratch_size, int* data,
                    int* reduce_locks, int* reduce_counts, const int nrecv_peers, int* sent_progress,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  assert(nloops <= MAX_NLOOPS);

  int pending_sends = 0;
  const int max_pending_sends = scratch_size / nelem_per_send;

  if (tid == 0) {
    assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
    assert(data_start % 4 == 0);
    assert(nelem_per_send % 4 == 0);
    assert((recv_sm || recv_proxy) == (nrecv_peers > 0));
    assert(!send_sm || !send_proxy);
    assert(!recv_sm || !recv_proxy);
  }

  __shared__ int ready;
  __shared__ int sent;

  if (tid == 0) {
    ready = (recv_sm || recv_proxy ? 0 : nloops);
    sent = (recv_sm || send_proxy ? 0 : nloops);

    if (!recv_sm && !recv_proxy && send_sm) {
      send_sm_channel->signal(nloops);
      sent = nloops;
    }
  }
  __syncthreads();

  int received = (recv_sm || recv_proxy ? 0 : nloops);
  int reduced = (recv_sm || recv_proxy ? 0 : nloops);

  while (reduced < nloops || sent < nloops) {
    if (received < nloops) {
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
          recv_sm_channel->get(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int), tid, blockDim.x);
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
      bool ready_to_send_sm = false; // only needed by thread 0
      if (recv_sm && nrecv_peers == 1) {
        // no __threadfence() needed, only one threadblock is writing
        for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
          int4 tmp = data4[offset];
          int4 val = recv_sm_channel->read<int4>(d_start4 + offset);
          tmp.w += val.w;
          tmp.x += val.x;
          tmp.y += val.y;
          tmp.z += val.z;
          data4[offset] = tmp;
        }
        if (nLastElem > 0 && tid == 0) {
          int4 tmp = data4[nElem4];
          int4 val = recv_sm_channel.read<int4>(d_start4 + nElem4);
          // assert 1 <= nLastElem <= 3
          tmp.w += val.w;
          if (nLastElem > 1) tmp.x += val.x;
          if (nLastElem > 2) tmp.y += val.y;
          data4[nElem4] = tmp;
        }
        ready_to_send_sm = true;
        ++reduced;
      } else {
        // Try lock
        __shared__ int lock_status;
        if (tid == 0) lock_status = atomicCAS(&reduce_locks[reduced], 0, 1);
        __syncthreads();
        if (!lock_status) {
          __threadfence();
          int4* const scratch4 = reinterpret_cast<int4*>(&recv_scratch[s_start]);
          for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
            int4 tmp = data4[offset];
            int4 val = scratch4[offset];
            tmp.w += val.w;
            tmp.x += val.x;
            tmp.y += val.y;
            tmp.z += val.z;
            data4[offset] = tmp;
          }
          if (nLastElem > 0 && tid == 0) {
            int4 tmp = data4[nElem4];
            int4 val = scratch4[nElem4];
            // assert 1 <= nLastElem <= 3
            tmp.w += val.w;
            if (nLastElem > 1) tmp.x += val.x;
            if (nLastElem > 2) tmp.y += val.y;
            data4[nElem4] = tmp;
          }

          // Unlock
          __syncthreads();
          if (tid == 0) {
            if (++reduce_counts[reduced] == nrecv_peers) ready_to_send_sm = true;
            reduce_locks[reduced] = 0;
            if (recv_proxy) {
              if (reduced > 0 && reduced % FLUSH_INTERVAL == 0) recv_proxy_channel->flush();
              recv_proxy_channel->signal();
            }
          }
          ++reduced;
        }
      }
      if (tid == 0 && ready_to_send_sm && send_sm) {
        send_sm_channel->signal();
        ++sent;
      }
      __syncthreads();
    }

    if (send_proxy) {
      if (tid == 0) {
        int sent_local = sent;
        if (sent_local < reduced) {
          bool ready_to_send_proxy = false;
          if (nrecv_peers == 1) {
            ready_to_send_proxy = true;
          } else {
            do {
              if (reduce_counts[sent_local] == nrecv_peers) {
                const int global_sent = atomicCAS(sent_progress, sent_local, sent_local + 1);
                if (global_sent == sent_local) {
                  ready_to_send_proxy = true;
                } else {
                  sent_local = global_sent;
                }
              }
            } while (!ready_to_send_proxy && sent_local < reduced);
          }
          if (ready_to_send_proxy) {
            if (pending_sends == max_pending_sends) {
              pending_sends -= send_proxy_channel->poll(pending_sends);
            }
            if (pending_sends < max_pending_sends) {
              const int64_t s_start = (sent_local % max_pending_sends) * nelem_per_send;
              const uint64_t d_start = data_start + sent_local * nelem_per_send;
              const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              if (sent_local > 0 && sent_local % FLUSH_INTERVAL == 0) send_proxy_channel->flush();
              send_proxy_channel->putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
              ++pending_sends;
              ++sent_local;
            }
          }
          sent = sent_local;
        }
      }
      __syncthreads();
    }
  }

}

__device__ mscclpp::DeviceSyncer deviceSyncer;

MSCCLPP_DEVICE_INLINE void zero_memory(int* data, const uint64_t nelem) {
  const int tid = threadIdx.x;
  int4* data4 = reinterpret_cast<int4*>(data);
  const uint64_t nElem4 = nElem / 4;
  const uint64_t nLastElem = nElem % 4;
  for (uint64_t off = tid; off < nElem4; off += blockDim.x) {
    int4 tmp = data4[offset];
    tmp.w = 0;
    tmp.x = 0;
    tmp.y = 0;
    tmp.z = 0;
    data4[offset] = tmp;
  }
  if (nLastElem > 0 && tid == 0) {
    int4 tmp = data4[nElem4];
    tmp.w = 0;
    if (nLastElem > 1) tmp.x = 0;
    if (nLastElem > 2) tmp.y = 0;
    data4[nElem4] = tmp;
  }
}

extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channel_block, mscclpp::SmChannelDeviceHandle* send_sm_channel_block,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channel_block, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channel_block,
           int* recv_sm_channel_indics, int* send_sm_channel_indics, int* recv_proxy_channel_indics, int* send_proxy_channel_indics,
           int** recv_scratch_block, const uint64_t scratch_size, int* data,
           int** reduce_locks_block, int** reduce_counts_block, int* nrecv_peers_block, int** sent_progress_block, bool* first_block,
           const uint64_t* data_start_block, const uint64_t nelem_per_send, const uint64_t* nelem_total_block) {
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
  int* reduce_locks = reduce_locks_block[bid];
  int* reduce_counts = reduce_counts_block[bid];
  const int nrecv_peers = nrecv_peers_block[bid];
  int* sent_progress = sent_progress_block[bid];
  const uint64_t data_start = data_start_block[bid];
  const uint64_t nelem_total = nelem_total_block[bid];

  if (is_first_block) {
    if (recv_sm || recv_proxy) {
      *sent_progress = 0;
      const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send;
      zero_memory(reduce_locks, nloops);
      zero_memory(reduce_counts, nloops);
    }
  }
  deviceSyncer.sync(gridDim.x);


  threadblockCall(recv_sm_channel, send_sm_channel, recv_proxy_channel, send_proxy_channel,
                  recv_scratch, recv_sm, send_sm, recv_proxy, send_proxy,
                  scratch_size, data,
                  reduce_locks, reduce_counts, nrecv_peers, sent_progress,
                  data_start, nelem_per_send, nelem_total);
}