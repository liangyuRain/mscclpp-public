// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_schedule
#define N_PEERS 8
#endif

// END_DEFINES //

/// The call is a single node in the tree.
///
/// Syncronization: It is guaranteed that in allreduce and allgather, after the kernel finishes,
/// the memory is safe to be written. However, the kernel immediately starts to write remote
/// memory once launched. There are two cases:
/// Allreduce - The kernel writes to remote scratch buffer in the beginning. As long as the remote
/// peer has finished initializing scratch buffer, the write is safe.
/// Allgather - The kernel writes to remote memory shard in the beginning. As long as the remote
/// peer has finished initializing all shards except its own, the write is safe. If the remote
/// needs to initialize all memory, then synchronization is needed to ensure remote initialization
/// finishes before local kernel launch.
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
/// @param data_start The data buffer start.
/// @param nelem_per_send Num of elements in each send.
/// @param nelem_total Total num of elements need to be send/recv.
/// @param scratch_size Max num of elements in scratch buffer for each recv channel (ignore if not reduce).
/// scratch_size must be greater than nelem_per_send
/// @param data Data buffer.
MSCCLPP_DEVICE_INLINE void 
    threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channels, mscclpp::SmChannelDeviceHandle* send_sm_channels,
                    mscclpp::SimpleProxyChannelDeviceHandle* recv_proxy_channels, mscclpp::SimpleProxyChannelDeviceHandle* send_proxy_channels,
                    int** recv_scratches, const int nsm_per_proxy,
                    const int nrecv_sm, const int nsend_sm, const int nrecv_proxy, const int nsend_proxy,
                    const char node_type, const uint64_t data_start,
                    const uint64_t nelem_per_send, const uint64_t nelem_total, const uint64_t scratch_size, int* data) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division
  const int nelem_per_send_sm = nelem_per_send / nsm_per_proxy;

  if (node_type <= 0) {
    int reduced[N_PEERS] = {};
    int sent[N_PEERS] = {};
    __shared__ int ready[N_PEERS];
    __shared__ int pending_sends[N_PEERS]; // Only reduce node needs to track pending sends.

    #pragma unroll
    for (int i = tid; i < N_PEERS; i += blockDim.x) {
      ready[i] = 0;
      pending_sends[i] = 0;
    }
    __syncthreads();

    const int max_pending_sends = scratch_size / nelem_per_send;

    int rloop = nrecv_sm + nrecv_proxy > 0 ? 0 : nloops; // progress of recv
    int sloop = nsend_sm + nsend_proxy > 0 ? 0 : nloops; // progress of send
    while (rloop < nloops || sloop < nloops) {
      if (rloop < nloops) {
        // assert nrecv_sm + nrecv_proxy > 0
        for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) {
          const int ready_loop = ready[i];
          // if (ready_loop < rloop + 1) ready[i] += (i < nrecv_sm ? recv_sm_channels[i].poll(rloop + 1 - ready_loop) :
          //                                                         recv_proxy_channels[i - nrecv_sm].poll(rloop + 1 - ready_loop));
          if (ready_loop < nloops) ready[i] += (i < nrecv_sm ? recv_sm_channels[i].poll(nloops - ready_loop) :
                                                               recv_proxy_channels[i - nrecv_sm].poll(nloops - ready_loop));
        }
        __syncthreads();

        int count[N_PEERS] = {};
        rloop = nloops;
        for (int i = 0; i < nrecv_sm + nrecv_proxy; ++i) {
          const int ready_loop = ready[i];
          if (reduced[i] < ready_loop){
            do {
              const bool sm_subchunk = (i < nrecv_sm && nsm_per_proxy > 1);
              const uint64_t s_start = (reduced[i] % max_pending_sends) * nelem_per_send + 
                  (sm_subchunk ? nelem_per_send_sm * (i % nsm_per_proxy) : 0);
              const uint64_t d_start = data_start + reduced[i] * nelem_per_send + 
                  (sm_subchunk ? nelem_per_send_sm * (i % nsm_per_proxy) : 0);
              const int diff = (sm_subchunk ? 1 : min(ready_loop - reduced[i], max_pending_sends - reduced[i] % max_pending_sends));

              const uint64_t nElem = (sm_subchunk ? nelem_per_send_sm, min(nelem_per_send * diff, data_start + nelem_total - d_start));
              const uint64_t nElem4 = nElem / 4;
              const uint64_t nLastElem = nElem % 4;

              int4* const data4 = (int4*) &data[d_start];
              int4* const scratch4 = (int4*) &recv_scratches[i][s_start];
              for (uint64_t offset = tid; offset < nElem4; offset += blockDim.x) {
                data4[offset].w += scratch4[offset].w;
                data4[offset].x += scratch4[offset].x;
                data4[offset].y += scratch4[offset].y;
                data4[offset].z += scratch4[offset].z;
              }
              for (uint64_t offset = tid; offset < nLastElem; offset += blockDim.x) {
                data[d_start + nElem4 * 4 + offset] += recv_scratches[i][s_start + nElem4 * 4 + offset];
              }
              reduced[i] += diff;
              count[i] += diff;
            } while (reduced[i] < ready_loop);
            __syncthreads();
          }
          if (reduced[i] < rloop) rloop = reduced[i];
        }

        for (int i = tid; i < nrecv_sm + nrecv_proxy; i += blockDim.x) {
          if (count[i] > 0) {
            if (i < nrecv_sm) recv_sm_channels[i].signal(count[i]);
            else recv_proxy_channels[i - nrecv_sm].signal(count[i]);
          }
        }
        __syncthreads(); // Necessary; otherwise, program can freeze in multirun allreduce.
      }

      if (sloop < rloop) {
        // assert nsend_sm + nsend_proxy > 0
        if (node_type == 0) { // root
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(rloop - sloop);
          for (int loop = sloop; loop < rloop; ++loop) {
            const uint64_t d_start = data_start + loop * nelem_per_send;
            const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
            for (int i = tid; i < nsend_proxy; i += blockDim.x) {
              send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
            }
          }
          sloop = rloop;
        } else {
          // assert (nsend_sm == nsm_per_proxy && nsend_proxy == 0) || (nsend_sm == 0 && nsend_proxy == 1)
          if (nsend_sm > 0) { // assert nsend_sm == nsm_per_proxy && nsend_proxy == 0
            bool readyToSend = false;
            for (int i = tid; i < nsm_per_proxy; i += blockDim.x) {
              int psends = pending_sends[i];
              int update = send_sm_channels[i].poll(psends);
              if (update > 0) {
                psends -= update;
                pending_sends[i] = psends;
              }
              if (psends < max_pending_sends && sent[i] < rloop) readyToSend = true;
            }
            if (!__syncthreads_or(readyToSend)) continue;

            sloop = rloop;
            for (int i = 0; i < nsm_per_proxy; ++i) {
              if (sent[i] == rloop) continue;
              int psends = pending_sends[i];
              if (psends < max_pending_sends) {
                do {
                  uint64_t s_start = (sent[i] % max_pending_sends) * nelem_per_send + nelem_per_send_sm * i;
                  uint64_t d_start = data_start + sent[i] * nelem_per_send + nelem_per_send_sm * i;
                  uint64_t size = min(nelem_per_send_sm, data_start + nelem_total - d_start);
                  send_sm_channels[i].put(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int), tid, blockDim.x);
                  ++psends;
                  ++sent[i];
                  __syncthreads();
                  if (tid == 0) send_sm_channels[i].signal();
                } while (psends < max_pending_sends && sent[i] < rloop);
                if (tid == 0) pending_sends[i] = psends;
              }
              if (sent[i] < sloop) sloop = sent[i];
            }
            __syncthreads();
          } else { // assert nsend_sm == 0 && nsend_proxy == 1
            int psends = pending_sends[0];
            if (psends == max_pending_sends) {
              if (tid == 0) pending_sends[0] -= send_proxy_channels[0].poll(psends);
              __syncthreads();
              psends = pending_sends[0];
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
              if (tid == 0) send_proxy_channels[0].putWithSignal(s_start * sizeof(int), d_start * sizeof(int), size * sizeof(int));
              ++sloop;
              ++psends;
            } while (psends < max_pending_sends && sloop < rloop);
            if (tid == 0) pending_sends[0] = psends;
            __syncthreads();
          }
        }
      }
    }
    if (node_type == 0) { // root
      for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].wait();
    }
  } else {
    if (nrecv_sm == 0 && nrecv_proxy == 0) {
      for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal(nloops);
      for (int sloop = 0; sloop < nloops; ++sloop) {
        const uint64_t d_start = data_start + sloop * nelem_per_send;
        const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
        for (int i = tid; i < nsend_proxy; i += blockDim.x) {
          send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
        }
      }
    } else { // assert (nrecv_sm == nsm_per_proxy && nrecv_proxy == 0) || (nrecv_sm == 0 && nrecv_proxy == 1)
      int sloop = 0;
      __shared__ int ready[N_PEERS];
      for (int i = tid; i < N_PEERS; i += blockDim.x) ready[i] = 0;
      __syncthreads();
      if (nrecv_sm > 0) { // assert (nrecv_sm == nsm_per_proxy && nrecv_proxy == 0)
        while (sloop < nloops) {
          
        }
      } else { // assert (nrecv_sm == 0 && nrecv_proxy == 1)

      }

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
          ++sloop;
          __syncthreads();
          for (int i = tid; i < nsend_proxy; i += blockDim.x)
            send_proxy_channels[i].putWithSignal(d_start * sizeof(int), size * sizeof(int));
          for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].signal();
        } while (sloop < ready_loop);
      }
    }
    if (tid == 0 && nrecv_sm == 1) recv_sm_channels[0].signal(); // `signal` to ensure sender wait until `get` finishes
    for (int i = tid; i < nsend_sm; i += blockDim.x) send_sm_channels[i].wait();
  }
  for (int i = tid; i < nsend_proxy; i += blockDim.x) send_proxy_channels[i].flush(); // question?
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
           char* node_types, uint64_t* data_start, const uint64_t nelem_per_send,
           uint64_t* nelem_total, const uint64_t scratch_size, int* data) {
  const int bid = blockIdx.x;

  threadblockCall(recv_sm_channels == nullptr ? nullptr : &recv_sm_channels[block_recv_sm_ch_starts[bid]], 
                  send_sm_channels == nullptr ? nullptr : &send_sm_channels[block_send_sm_ch_starts[bid]],
                  recv_proxy_channels == nullptr ? nullptr : &recv_proxy_channels[block_recv_proxy_ch_starts[bid]], 
                  send_proxy_channels == nullptr ? nullptr : &send_proxy_channels[block_send_proxy_ch_starts[bid]],
                  &recv_scratches[block_scratch_starts[bid]], nrecvs_sm[bid], nsends_sm[bid], nrecvs_proxy[bid], nsends_proxy[bid],
                  node_types[bid], data_start[bid], nelem_per_send, nelem_total[bid], scratch_size, data);
}