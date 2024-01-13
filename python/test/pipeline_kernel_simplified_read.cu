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

#define FLUSH_INTERVAL 100

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
    int pending_sends = 0; // Only reduce node's proxy channel needs to track pending sends.
                           // Only thread(tid=0) needs to track pending sends.

    #pragma unroll
    for (int i = tid; i < N_PEERS; i += blockDim.x) ready[i] = 0;
    __syncthreads();

    const int max_pending_sends = scratch_size / nelem_per_send;

    int min_ready = 0;
    for (int loop = 0; loop < nloops; ++loop) {
      if (nrecv_sm + nrecv_proxy > 0) {
        const uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
        const uint64_t d_start = data_start + loop * nelem_per_send;
        const uint64_t d_start4 = d_start / 4;
        const uint64_t nElem = min(nelem_per_send, data_start + nelem_total - d_start);
        const uint64_t nElem4 = nElem / 4;
        const uint64_t nLastElem = nElem % 4;
        int4* const data4 = (int4*) &data[d_start];

        do {
          if (min_ready <= loop) {
            bool hasUpdate;
            do {
              hasUpdate = false;
              for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) {
                const int ready_loop = ready[i];
                if (ready_loop < nloops) {
                  int update = (i < nrecv_sm ? recv_sm_channels[i].poll(nloops - ready_loop) :
                                               recv_proxy_channels[i - nrecv_sm].poll(nloops - ready_loop));
                  if (update > 0) hasUpdate = true;
                  ready[i] = ready_loop + update;
                }
              }
            } while (!__syncthreads_or(hasUpdate));
          }

          min_ready = nloops;
          bool chHasUpdate[N_PEERS];
          for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
            const int ready_loop = ready[i];
            if (ready_loop < min_ready) min_ready = ready_loop;
            if (ready_loop > loop && reduced[i] == loop) {
              chHasUpdate[i] = true;
              ++reduced[i];
            } else {
              chHasUpdate[i] = false;
            }
          }
          for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
            int4 tmp = data4[offset];
            for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
              if (chHasUpdate[i]) {
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
            assert(loop == nloops - 1);
            int4 tmp = data4[nElem4];
            for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
              if (chHasUpdate[i]) {
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
          __syncthreads();
          // sm_proxy_channels do not need to be signaled, because instead of remote writes to
          // local scratch buffer, we read from remote memory directly.
          for (int i = tid; i < nrecv_proxy; i += blockDim.x) {
            if (chHasUpdate[i + nrecv_sm]) {
              if (loop > 0 && loop % FLUSH_INTERVAL == 0) recv_proxy_channels[i].flush();
              recv_proxy_channels[i].signal();
            }
          }
          __syncthreads();
        } while (min_ready <= loop);
      }

      if (nsend_sm + nsend_proxy > 0) {
        if (node_type == 0) { // root
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal();
          for (int i = tid; i < nsend_proxy; i += blockDim.x) {
            uint64_t d_start = data_start + loop * nelem_per_send;
            uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
            if (loop == 0) send_proxy_channels[i].wait();
            else if (loop % FLUSH_INTERVAL == 0) send_proxy_channels[i].flush();
            send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
          }
        } else {
          // assert nsend_sm + nsend_proxy == 1
          if (tid == 0) {
            if (nsend_sm == 1) {
              send_sm_channels[0].signal();
            } else {
              if (pending_sends == max_pending_sends) {
                send_proxy_channels[0].wait();
                pending_sends -= 1 + send_proxy_channels[0].poll(pending_sends - 1);
              }

              uint64_t s_start = (loop % max_pending_sends) * nelem_per_send;
              uint64_t d_start = data_start + loop * nelem_per_send;
              uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              if (loop > 0 && loop % FLUSH_INTERVAL == 0) send_proxy_channels[0].flush();
              send_proxy_channels[0].putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
              ++pending_sends;
            }
          }
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
        uint64_t d_start = data_start + sloop * nelem_per_send;
        uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
        for (int i = tid; i < nsend_proxy; i += blockDim.x) {
          if (sloop == 0) send_proxy_channels[i].wait();
          send_proxy_channels[i].putWithSignalAndFlush(d_start * sizeof(int), size * sizeof(int));
        }
      }
    } else {
      if (tid == 0 && nrecv_proxy == 1) recv_proxy_channels[0].signal();
      int sloop = 0;
      __shared__ int ready;
      if (tid == 0) ready = 0;
      while (sloop < nloops) {
        if (tid == 0) {
          if (nrecv_sm == 1) {
            recv_sm_channels[0].wait();
            ready = sloop + 1 + recv_sm_channels[0].poll(nloops - sloop - 1);
          } else {
            recv_proxy_channels[0].wait();
            ready = sloop + 1 + recv_proxy_channels[0].poll(nloops - sloop - 1);
          }
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