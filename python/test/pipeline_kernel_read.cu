// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <assert.h>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_schedule
#define N_PEERS 8
#endif

#define FLUSH_INTERVAL 50

// END_DEFINES //

/// The call is a single node in the tree.
///
/// Syncronization: It is guaranteed that in allreduce and allgather, after the kernel finishes,
/// the memory is safe to be written. However, the kernel immediately starts to write remote
/// memory once launched. There are two cases:
/// Allreduce/ReduceScatter - The kernel writes to remote scratch buffer in the beginning.
/// As long as the remote peer has finished initializing scratch buffer, the write is safe.
/// Allgather - The kernel (proxy) waits for remote signal before writing to remote.
///
/// @param recv_sm_channels SM channels for recv.
/// @param send_sm_channels SM channels for send.
/// @param recv_proxy_channels Proxy channels for recv.
/// @param send_proxy_channels Proxy channels for send.
/// @param recv_scratches Scratch buffers for each recv_sm_channels, len(recv_scratches) == len(recv_sm_channels) + len(recv_proxy_channels)
/// @param nrecv_sm Num of valid recv_sm_channels (nrecv_sm + nrecv_proxy <= 1 if broadcast).
/// @param nsend_sm Num of valid send_sm_channels (nsend_sm + nsend_proxy <= 1 if reduce).
/// @param nrecv_proxy Num of valid recv_proxy_channels (nrecv_sm + nrecv_proxy <= 1 if broadcast).
/// @param nsend_proxy Num of valid send_proxy_channels (nsend_sm + nsend_proxy <= 1 if reduce).
/// @param node_type <0: reduce node; =0: root node; >0: broadcast node.
/// The send channels of broadcast and root nodes write to data buffer.
/// The send channels of reduce node write to scratch buffer.
/// @param scratch_size Max num of elements in scratch buffer for each recv channel (ignore if not reduce).
/// scratch_size must be greater than nelem_per_send
/// @param data Data buffer.
/// @param data_start The data buffer start.
/// @param nelem_per_send Num of elements in each send.
/// @param nelem_total Total num of elements need to be send/recv.
MSCCLPP_DEVICE_INLINE void 
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channels, mscclpp::SmChannelDeviceHandle* send_sm_channels,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
                    int** recv_scratches, const int nrecv_sm, const int nsend_sm, const int nrecv_proxy, const int nsend_proxy,
                    const char node_type, const uint64_t scratch_size, int* data,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division

  if (node_type <= 0) {
    if (tid == 0) {
      assert(reinterpret_cast<uintptr_t>(data) % alignof(int4) == 0);
      assert(data_start % 4 == 0);
      assert(nelem_per_send % 4 == 0);
    }

    int reduced[N_PEERS] = {};
    __shared__ int ready[N_PEERS];
    __shared__ int shared_sloop; // used to sync sloop across threads if not root
    int pending_sends = 0; // Only reduce node's proxy channel needs to track pending sends.
                           // Only thread(tid=0) needs to track pending sends.
    int poll_loop_cnt = 0; // Only thread(tid=0) needs to track poll_loop_cnt.

    #pragma unroll
    for (int i = tid; i < N_PEERS; i += blockDim.x) ready[i] = 0;
    if (tid == 0) shared_sloop = (nsend_sm + nsend_proxy > 0 ? 0 : nloops);
    __syncthreads();

    const int max_pending_sends = scratch_size / nelem_per_send;

    int rloop = nrecv_sm + nrecv_proxy > 0 ? 0 : nloops; // progress of recv
    int sloop = nsend_sm + nsend_proxy > 0 ? 0 : nloops; // progress of send
    while (rloop < nloops || sloop < nloops) {
      if (rloop < nloops) {
        // assert nrecv_sm + nrecv_proxy > 0
        bool ready_for_reduce = false;
        for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) {
          int ready_loop = ready[i];
          // if (ready_loop < rloop + 1) {
          //   const int update = (i < nrecv_sm ? recv_sm_channels[i].poll(rloop + 1 - ready_loop) :
          //                                      recv_proxy_channels[i - nrecv_sm].poll(rloop + 1 - ready_loop));
          if (ready_loop < nloops) {
            const int update = (i < nrecv_sm ? recv_sm_channels[i].poll(nloops - ready_loop) :
                                               recv_proxy_channels[i - nrecv_sm].poll(nloops - ready_loop));
            if (update > 0) {
              ready_loop += update;
              ready[i] = ready_loop;
            }
          }
          if (ready_loop > reduced[i]) ready_for_reduce = true;
        }

        if (__syncthreads_or(ready_for_reduce)) {
          // assert rloop == min(reduced[*])
          int count[N_PEERS] = {};
          bool ready_to_send = true;
          int min_ready = nloops + 1; // min unreduced but ready
          int max_ready = -1; // max unreduced but ready
          for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
            const int ready_loop = ready[i];
            // assert rloop == min(reduced[*]) <= ready_loop
            // Note sloop = nloops if nsend_sm + nsend_proxy == 0
            if (ready_loop <= sloop) ready_to_send = false;
            if ((count[i] = ready_loop - reduced[i]) > 0) {
              if (reduced[i] < min_ready) min_ready = reduced[i];
              if (ready_loop > max_ready) max_ready = ready_loop;
            }
          }
          if (max_ready > min_ready) {
            // Note that ready_to_send does not imply there is data ready for reduce.
            if (ready_to_send && max_ready > min_ready) {
              assert(min_ready >= rloop); // min_ready may be larger than rloop: 
                                          // the peer with ready_loop == rloop may not signal yet.
              max_ready = min_ready + 1; // reduce and send immediately
            }
            for (int k = min_ready; k < max_ready; ++k) {
              const uint64_t s_start = (k % max_pending_sends) * nelem_per_send;
              const uint64_t d_start = data_start + k * nelem_per_send;
              const uint64_t d_start4 = d_start / 4;
              const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
              const uint64_t nElem4 = nElem / 4;
              const uint64_t nLastElem = nElem % 4;
              int4* const data4 = (int4*) &data[d_start];
              for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
                int4 tmp = data4[offset];
                for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
                  if (reduced[i] <= k && k < reduced[i] + count[i]) {
                    int4 val;
                    if (i < nrecv_sm) val = recv_sm_channels[i].read<int4>(d_start4 + offset);
                    else val = reinterpret_cast<int4*>(&recv_scratches[i][s_start])[offset];
                    tmp.w += val.w;
                    tmp.x += val.x;
                    tmp.y += val.y;
                    tmp.z += val.z;
                  }
                }
                data4[offset] = tmp;
              }
              if (nLastElem > 0 && tid == 0) {
                assert(k == nloops - 1);
                int4 tmp = data4[nElem4];
                for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
                  if (reduced[i] <= k && k < reduced[i] + count[i]) {
                    int4 val;
                    if (i < nrecv_sm) val = recv_sm_channels[i].read<int4>(d_start4 + nElem4);
                    else val = reinterpret_cast<int4*>(&recv_scratches[i][s_start])[nElem4];
                    // assert 1 <= nLastElem <= 3
                    tmp.w += val.w;
                    if (nLastElem > 1) tmp.x += val.x;
                    if (nLastElem > 2) tmp.y += val.y;
                  }
                }
                data4[nElem4] = tmp;
              }
            }
            __syncthreads();
            rloop = nloops;
            for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
              count[i] = min(count[i], max(0, max_ready - reduced[i]));
              reduced[i] += count[i];
              if (reduced[i] < rloop) rloop = reduced[i];
            }
            // sm_channels do not need to be signaled, because instead of remote writes to
            // local scratch buffer, we read from remote memory directly.
            for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) {
              // assert if min_ready >= max_ready, then min_ready = nloops + 1 and max_ready = -1.
              if (count[i] > 0) {
                if (i >= nrecv_sm) {
                  const int before = reduced[i] - count[i];
                  if (before > 0 && before / FLUSH_INTERVAL < reduced[i] / FLUSH_INTERVAL) recv_proxy_channels[i - nrecv_sm].flush();
                  recv_proxy_channels[i - nrecv_sm].signal(count[i]);
                }
              }
            }
          }
          __syncthreads(); // Necessary; otherwise, program can freeze in multirun allreduce.
        }
      }

      if (nsend_proxy == 1 && rloop < nloops) {
        if (tid == 0) {
          if (pending_sends == 0) {
            poll_loop_cnt = 0;
          } else {
            ++poll_loop_cnt;
            if (poll_loop_cnt >= 10) {
              pending_sends -= send_proxy_channels[0].poll(pending_sends);
              poll_loop_cnt = 0;
            }
          }
        }
        __syncthreads();
      }

      if (sloop < rloop) {
        // assert nsend_sm + nsend_proxy > 0
        if (node_type == 0) { // root
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(rloop - sloop);
          for (int loop = sloop; loop < rloop; ++loop) {
            const uint64_t d_start = data_start + loop * nelem_per_send;
            const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
            for (int i = tid; i < nsend_proxy; i += blockDim.x) {
              if (loop == 0) send_proxy_channels[i].wait();
              else if (loop % FLUSH_INTERVAL == 0) send_proxy_channels[i].flush();
              send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
            }
          }
          sloop = rloop;
        } else {
          if (tid == 0) {
            if (nsend_sm == 1) {
              send_sm_channels[0].signal(rloop - sloop);
              sloop = rloop;
            } else {
              if (pending_sends == max_pending_sends) {
                pending_sends -= send_proxy_channels[0].poll(pending_sends);
                poll_loop_cnt = 0;
              }
              while (pending_sends < max_pending_sends && sloop < rloop) {
                uint64_t s_start = (sloop % max_pending_sends) * nelem_per_send;
                uint64_t d_start = data_start + sloop * nelem_per_send;
                uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
                if (sloop > 0 && sloop % FLUSH_INTERVAL == 0) send_proxy_channels[0].flush();
                send_proxy_channels[0].putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
                ++sloop;
                ++pending_sends;
              }
            }
            shared_sloop = sloop;
          }
          __syncthreads();
          sloop = shared_sloop;
          __syncthreads();
        }
      }
    }
    for (int i = tid; i < nrecv_sm; i += blockDim.x) recv_sm_channels[i].signal();
    if (node_type == 0) { // root
      for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].wait();
    } else {
      if (tid == 0) {
        if (nsend_sm == 1) send_sm_channels[0].wait();
        else if (nsend_proxy == 1) {
          do {
            pending_sends -= send_proxy_channels[0].poll(pending_sends);
          } while (pending_sends > 0);
        }
      }
    }
  } else {
    // assert nrecv_sm + nrecv_proxy <= 1
    if (nrecv_sm == 0 && nrecv_proxy == 0) {
      for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(nloops);
      for (int sloop = 0; sloop < nloops; ++sloop) {
        const uint64_t d_start = data_start + sloop * nelem_per_send;
        const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
        for (int i = tid; i < nsend_proxy; i += blockDim.x) {
          if (sloop == 0) send_proxy_channels[i].wait();
          else send_proxy_channels[i].flush();
          send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
        }
      }
    } else {
      if (tid == 0 && nrecv_proxy == 1) recv_proxy_channels[0].signal();
      int sloop = 0;
      __shared__ int ready;
      if (tid == 0) ready = 0;
      while (sloop < nloops) {
        if (tid == 0) {
          int ready_loop = sloop;
          if (nrecv_sm == 1) {
            do {
              ready_loop += recv_sm_channels[0].poll(nloops - ready_loop);
            } while (ready_loop == sloop);
          } else {
            do {
              ready_loop += recv_proxy_channels[0].poll(nloops - ready_loop);
            } while (ready_loop == sloop);
          }
          ready = ready_loop;
        }
        __syncthreads();
        const int ready_loop = ready;
        do {
          uint64_t d_start = data_start + sloop * nelem_per_send;
          uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
          if (nrecv_sm == 1) recv_sm_channels[0].get(d_start * sizeof(int), size * sizeof(int), tid, blockDim.x);
          __syncthreads();
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal();
          for (int i = tid; i < nsend_proxy; i += blockDim.x) {
            if (sloop == 0) send_proxy_channels[i].wait();
            else if (sloop % FLUSH_INTERVAL == 0) send_proxy_channels[i].flush();
            send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
          }
          ++sloop;
        } while (sloop < ready_loop);
      }
    }
    if (tid == 0 && nrecv_sm == 1) recv_sm_channels[0].signal(); // `signal` to ensure sender wait until `get` finishes
    for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].wait();
  }
  for (int i = tid; i < nrecv_proxy; i += blockDim.x) recv_proxy_channels[i].flush();
  for (int i = tid; i < nsend_proxy; i += blockDim.x) send_proxy_channels[i].flush();
}

/// Call threadblockCall.
/// SM channel scratches: recv_scratches[block_scratch_starts[bid], ..., block_scratch_starts[bid] + nrecvs_sm[bid] - 1]
/// Proxy channel scratches: recv_scratches[block_scratch_starts[bid] + nrecvs_sm[bid], ..., block_scratch_starts[bid] + nrecvs_sm[bid] + nrecvs_proxy[bid] - 1]
extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channels, mscclpp::SmChannelDeviceHandle* send_sm_channels,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
           int** recv_scratches, int* block_recv_sm_ch_starts, int* block_send_sm_ch_starts,
           int* block_recv_proxy_ch_starts, int* block_send_proxy_ch_starts,
           int* block_scratch_starts,
           int* nrecvs_sm, int* nsends_sm, int* nrecvs_proxy, int* nsends_proxy,
           char* node_types, const uint64_t scratch_size, int* data, 
           const uint64_t* data_start, const uint64_t nelem_per_send, const uint64_t* nelem_total) {
  const int bid = blockIdx.x;

  threadblockCall(recv_sm_channels == nullptr ? nullptr : &recv_sm_channels[block_recv_sm_ch_starts[bid]], 
                  send_sm_channels == nullptr ? nullptr : &send_sm_channels[block_send_sm_ch_starts[bid]],
                  recv_proxy_channels == nullptr ? nullptr : &recv_proxy_channels[block_recv_proxy_ch_starts[bid]], 
                  send_proxy_channels == nullptr ? nullptr : &send_proxy_channels[block_send_proxy_ch_starts[bid]],
                  &recv_scratches[block_scratch_starts[bid]], nrecvs_sm[bid], nsends_sm[bid], nrecvs_proxy[bid], nsends_proxy[bid],
                  node_types[bid], scratch_size, data,
                  data_start[bid], nelem_per_send, nelem_total[bid]);
}