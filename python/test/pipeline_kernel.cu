// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_schedule
#define N_PEERS 8
#define TD int
#endif

// END_DEFINES //

/// The call is a single node in the tree.
///
/// @param recv_sm_channels SM channels for recv.
/// @param send_sm_channels SM channels for send.
/// @param recv_scratches Scratch buffers for each recv_sm_channels, len(recv_scratches) == len(recv_sm_channels)
/// @param nrecv_sm Num of valid recv_sm_channels (must <= 1 if not reduce).
/// @param nsend_sm Num of valid send_sm_channels.
/// @param node_type <0: reduce node; =0: root node; >0: broadcast node.
/// The send channels of broadcast and root nodes write to data buffer.
/// The send channels of reduce node write to scratch buffer.
/// @param data_start The data buffer start.
/// @param nelem_per_send Num of elements in each send.
/// @param nelem_total Total num of elements need to be send/recv.
/// @param scratch_size Max num of elements in scratch buffer for each recv channel (ignore if not reduce).
/// scratch_size must be greater than nelem_per_send
/// @param data Data buffer.
MSCCLPP_DEVICE_INLINE void 
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channels, mscclpp::SmChannelDeviceHandle* send_sm_channels,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
                    TD** recv_scratches, const int nrecv_sm, const int nsend_sm, const int nrecv_proxy, const int nsend_proxy,
                    const char node_type, const uint64_t data_start,
                    const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t scratch_size, TD* data) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division

  if (node_type <= 0) {
    int reduced[N_PEERS] = {};
    __shared__ int ready[N_PEERS];
    __shared__ int pending_sends; // Only reduce node needs to track pending sends.
                                  // Reduce node has at most one send peer.

    #pragma unroll
    for (int i = tid; i < N_PEERS; i += blockDim.x) ready[i] = 0;
    if (tid == 0) pending_sends = 0;
    __syncthreads();

    const int max_pending_sends = scratch_size / nelem_per_send;

    int rloop = nrecv_sm + nrecv_proxy > 0 ? 0 : nloops; // progress of recv
    int sloop = nsend_sm + nsend_proxy > 0 ? 0 : nloops; // progress of send
    while (rloop < nloops || sloop < nloops) {
      if (rloop < nloops) {
        // assert nrecv_sm > 0
        for (int i = tid; i < nrecv_sm; i += blockDim.x) {
          const int ready_loop = ready[i];
          // if (ready_loop < rloop + 1) ready[i] += recv_sm_channels[i].poll(rloop + 1 - ready_loop);
          if (ready_loop < nloops) ready[i] += recv_sm_channels[i].poll(nloops - ready_loop);
        }
        for (int i = tid; i < nrecv_proxy; i += blockDim.x) {
          const int peerIdx = i + nrecv_sm;
          const int ready_loop = ready[peerIdx];
          // if (ready_loop < rloop + 1) ready[peerIdx] += recv_proxy_channels[i].poll(rloop + 1 - ready_loop);
          if (ready_loop < nloops) ready[peerIdx] += recv_proxy_channels[i].poll(nloops - ready_loop);
        }
        __syncthreads();

        int count[N_PEERS] = {};
        rloop = nloops;
        for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
          const int ready_loop = ready[i];
          if (reduced[i] < ready_loop){
            do {
              uint64_t s_start = (reduced[i] % max_pending_sends) * nelem_per_send;
              uint64_t d_start = data_start + reduced[i] * nelem_per_send;
              int diff = min(ready_loop - reduced[i], max_pending_sends - reduced[i] % max_pending_sends);
              for (uint64_t offset = tid;
                   offset < nelem_per_send * diff && d_start + offset < data_start + nelem_total;
                   offset += blockDim.x) {
                data[d_start + offset] += recv_scratches[i][s_start + offset];
              }
              reduced[i] += diff;
              count[i] += diff;
            } while (reduced[i] < ready_loop);
            __syncthreads();
          }
          if (reduced[i] < rloop) rloop = reduced[i];
        }

        for (int i = tid; i < nrecv_sm; i += blockDim.x) {
          if (count[i] > 0) recv_sm_channels[i].signal(count[i]);
        }
        for (int i = tid; i < nrecv_proxy; i += blockDim.x) {
          const int peerIdx = i + nrecv_sm;
          if (count[peerIdx] > 0) recv_proxy_channels[i].signal(count[peerIdx]);
        }
      }

      if (sloop < rloop) {
        // assert nsend_sm + nsend_proxy > 0
        if (node_type == 0) { // root
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(rloop - sloop);
          for (int i = tid; i < nsend_proxy; i += blockDim.x) {
            for (int loop = sloop; loop < rloop; ++loop) {
              uint64_t d_start = data_start + loop * nelem_per_send;
              uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              send_proxy_channels[i].putWithSignal(d_start * sizeof(TD), size * sizeof(TD));
            }
          }
          sloop = rloop;
        } else {
          // assert nsend_sm + nsend_proxy == 1
          int psends = pending_sends;
          if (psends == max_pending_sends) {
            if (tid == 0) {
              if (nsend_sm == 1) pending_sends -= send_sm_channels[0].poll(psends);
              else if (nsend_proxy == 1) pending_sends -= send_proxy_channels[0].poll(psends);
            }
            __syncthreads();
            psends = pending_sends;
            if (psends == max_pending_sends) {
              __syncthreads();
              continue;
            }
          }
          
          // pipeline send: ensure one send (one nelem_per_send) one signal
          if (nsend_sm == 1) {
            do {
              uint64_t s_start = (sloop % max_pending_sends) * nelem_per_send;
              uint64_t d_start = data_start + sloop * nelem_per_send;
              uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              send_sm_channels[0].put(s_start * sizeof(TD), d_start * sizeof(TD), size * sizeof(TD), tid, blockDim.x);
              ++sloop;
              ++psends;
              __syncthreads();
              if (tid == 0) send_sm_channels[0].signal();
            } while (psends < max_pending_sends && sloop < rloop);
          } else if (nsend_proxy == 1) {
            do {
              uint64_t s_start = (sloop % max_pending_sends) * nelem_per_send;
              uint64_t d_start = data_start + sloop * nelem_per_send;
              uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
              if (tid == 0) send_proxy_channels[0].putWithSignal(s_start * sizeof(TD), d_start * sizeof(TD), size * sizeof(TD));
              ++sloop;
              ++psends;
            } while (psends < max_pending_sends && sloop < rloop);
          }
          if (tid == 0) pending_sends = psends;
          __syncthreads();
        }
      }
    }
  } else {
    // assert nrecv_sm + nrecv_sm <= 1
    if (nrecv_sm == 0 && nrecv_sm == 0) {
      for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(nloops);
      for (int i = tid; i < nsend_proxy; i += blockDim.x) {
        for (int sloop = 0; sloop < nloops; ++sloop) {
          uint64_t d_start = data_start + sloop * nelem_per_send;
          uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
          send_proxy_channels[i].putWithSignal(d_start * sizeof(TD), size * sizeof(TD));
        }
      }
    } else {
      int sloop = 0;
      __shared__ int ready;
      if (tid == 0) ready = 0;
      while (sloop < nloops) {
        if (tid == 0) {
          int ready_loop = sloop;
          do {
            if (nrecv_sm == 1) ready_loop += recv_sm_channels[0].poll(nloops - ready_loop);
            else if (nrecv_proxy == 1) ready_loop += recv_proxy_channels[0].poll(nloops - ready_loop);
          } while (ready_loop == sloop);
          ready = ready_loop;
        }
        __syncthreads();
        const int ready_loop = ready;
        do {
          uint64_t d_start = data_start + sloop * nelem_per_send;
          uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
          if (nrecv_sm == 1) recv_sm_channels[0].get(d_start * sizeof(TD), size * sizeof(TD), tid, blockDim.x);
          ++sloop;
          __syncthreads();
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal();
          for (int i = tid; i < nsend_proxy; i += blockDim.x) {
            send_sm_channels[i].putWithSignal(d_start * sizeof(TD), size * sizeof(TD));
          }
        } while (sloop < ready_loop);
      }
    }
  }
}

/// Call threadblockCall.
/// SM channel scratches: recv_scratches[block_scratch_starts[bid], ..., block_scratch_starts[bid] + nrecvs_sm[bid] - 1]
/// Proxy channel scratches: recv_scratches[block_scratch_starts[bid] + nrecvs_sm[bid], ..., block_scratch_starts[bid] + nrecvs_sm[bid] + nrecvs_proxy[bid] - 1]
extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_sm_channels,
           mscclpp::SmChannelDeviceHandle* send_sm_channels,
           mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels,
           mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
           TD** recv_scratches, int* block_recv_sm_ch_starts, int* block_send_sm_ch_starts,
           int* block_recv_proxy_ch_starts, int* block_send_proxy_ch_starts,
           int* block_scratch_starts,
           int* nrecvs_sm, int* nsends_sm, int* nrecvs_proxy, int* nsends_proxy,
           char* node_types, uint64_t* data_start, const uint64_t nelem_per_send,
           uint64_t* nelem_total, const uint64_t scratch_size, TD* data) {
  const int bid = blockIdx.x;

  threadblockCall(&recv_sm_channels[block_recv_sm_ch_starts[bid]], &send_sm_channels[block_send_sm_ch_starts[bid]],
                  &recv_proxy_channels[block_recv_proxy_ch_starts[bid]], &send_proxy_channels[block_send_proxy_ch_starts[bid]],
                  &recv_scratches[block_scratch_starts[bid]], nrecvs_sm[bid], nsends_sm[bid], nrecvs_proxy[bid], nsends_proxy[bid],
                  node_types[bid], data_start[bid], nelem_per_send, nelem_total[bid], scratch_size, data);
}