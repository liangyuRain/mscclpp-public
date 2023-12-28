// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_schedule
#define N_PEERS 8
#define N_RECV_CHANNELS 1
#define N_SEND_CHANNELS 1
#define TD int
#endif

// END_DEFINES //

/// The call is a single node in the tree.
///
/// @param recv_channels SM channels for recv.
/// @param send_channels SM channels for send.
/// @param recv_scratches Scratch buffers for each recv_channels, len(recv_scratches) == len(recv_channels)
/// @param nrecv Num of valid recv_channels (must <= 1 if not reduce).
/// @param nsend Num of valid send_channels.
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
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_channels, mscclpp::SmChannelDeviceHandle* send_channels,
                    TD** recv_scratches, const int nrecv, const int nsend, const char node_type, const uint64_t data_start,
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

    int rloop = nrecv > 0 ? 0 : nloops; // progress of recv
    int sloop = nsend > 0 ? 0 : nloops; // progress of send
    while (rloop < nloops || sloop < nloops) {
      if (rloop < nloops) {
        // assert nrecv > 0
        for (int i = tid; i < nrecv; i += blockDim.x) {
          const int ready_loop = ready[i];
          // if (ready_loop < rloop + 1) ready[i] += recv_channels[i].poll(rloop + 1 - ready_loop);
          if (ready_loop < nloops) ready[i] += recv_channels[i].poll(nloops - ready_loop);
        }
        __syncthreads();

        int count[N_PEERS] = {};
        rloop = nloops;
        for (int i = 0; i < nrecv; ++i) {
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

        for (int i = tid; i < nrecv; i += blockDim.x) {
          if (count[i] > 0) recv_channels[i].signal(count[i]);
        }
      }

      if (sloop < rloop) {
        // assert nsend > 0
        if (node_type == 0) { // root
          for (int i = tid; i < nsend; i += blockDim.x) send_channels[i].signal(rloop - sloop);
          sloop = rloop;
        } else {
          // assert nsend == 1
          int psends = pending_sends;
          if (psends == max_pending_sends) {
            if (tid == 0) pending_sends -= send_channels[0].poll(psends);
            __syncthreads();
            psends = pending_sends;
            if (psends == max_pending_sends) {
              __syncthreads();
              continue;
            }
          }
          
          // pipeline send: ensure one send (one nelem_per_send) one signal
          do {
            uint64_t s_start = (sloop % max_pending_sends) * nelem_per_send;
            uint64_t d_start = data_start + sloop * nelem_per_send;
            uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
            send_channels[0].put(s_start * sizeof(TD), d_start * sizeof(TD), size * sizeof(TD), tid, blockDim.x);
            ++sloop;
            ++psends;
            __syncthreads();
            if (tid == 0) send_channels[0].signal();
          } while (psends < max_pending_sends && sloop < rloop);
          if (tid == 0) pending_sends = psends;
          __syncthreads();
        }
      }
    }
  } else {
    // assert nrecv <= 1
    if (nrecv == 0) {
      for (int i = tid; i < nsend; i += blockDim.x) send_channels[i].signal(nloops);
    } else {
      int sloop = 0;
      __shared__ int ready;
      if (tid == 0) ready = 0;
      while (sloop < nloops) {
        if (tid == 0) {
          int ready_loop = sloop;
          do {
            ready_loop += recv_channels[0].poll(nloops - ready_loop);
          } while (ready_loop == sloop);
          ready = ready_loop;
        }
        __syncthreads();
        const int ready_loop = ready;
        do {
          uint64_t d_start = data_start + sloop * nelem_per_send;
          uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
          recv_channels[0].get(d_start * sizeof(TD), size * sizeof(TD), tid, blockDim.x);
          ++sloop;
          __syncthreads();
          for (int i = tid; i < nsend; i += blockDim.x) send_channels[i].signal();
        } while (sloop < ready_loop);
      }
    }
  }
}

/// Call threadblockCall.
extern "C" __global__ void __launch_bounds__(1024)
    KERNEL(mscclpp::SmChannelDeviceHandle* recv_channels,
           mscclpp::SmChannelDeviceHandle* send_channels,
           TD** recv_scratches, int* block_recv_ch_starts, int* block_send_ch_starts,
           int* nrecvs, int* nsends, char* node_types, uint64_t* data_start, const uint64_t nelem_per_send,
           uint64_t* nelem_total, const uint64_t scratch_size, TD* data) {
  const int bid = blockIdx.x;

  threadblockCall(&recv_channels[block_recv_ch_starts[bid]], &send_channels[block_send_ch_starts[bid]],
                  &recv_scratches[block_recv_ch_starts[bid]], nrecvs[bid], nsends[bid], node_types[bid],
                  data_start[bid], nelem_per_send, nelem_total[bid], scratch_size, data);
}