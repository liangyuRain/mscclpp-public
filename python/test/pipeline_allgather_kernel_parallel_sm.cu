// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <mscclpp/proxy_channel_device.hpp>

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL pipeline_schedule
#define N_PEERS 8
#endif

#define FLUSH_INTERVAL 50

// END_DEFINES //

static_assert(sizeof(mscclpp::DeviceSyncer) == 12, "sizeof(mscclpp::DeviceSyncer) != 12");


MSCCLPP_DEVICE_INLINE void
    parallel_sm_threadblockCall(mscclpp::SmChannelDeviceHandle* recv_sm_channel,
                                const int sm_block_idx, const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer,
                                const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total) {
  const int tid = threadIdx.x;
  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division

  for (int sloop = 0; sloop < nloops; ++sloop) {
    const uint64_t d_start = data_start + sloop * nelem_per_send;
    const uint64_t size = min(nelem_per_send, data_start + nelem_total - d_start);
    sm_syncer->sync(sm_block_cnt);
    recv_sm_channel->get(d_start * sizeof(int), size * sizeof(int),
                         tid + sm_block_idx * blockDim.x, sm_block_cnt * blockDim.x);
    sm_syncer->sync(sm_block_cnt);
  }
}

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
                    const int sm_block_cnt, mscclpp::DeviceSyncer* sm_syncer,
                    const uint64_t data_start, const uint64_t nelem_per_send, const uint64_t nelem_total) {
  const int tid = threadIdx.x;

  const int nloops = (nelem_total + nelem_per_send - 1) / nelem_per_send; // ceiling division

  assert(node_type > 0);

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
        if (nrecv_sm == 1) {
          sm_syncer->sync(sm_block_cnt);
          recv_sm_channels[0].get(d_start * sizeof(int), size * sizeof(int), tid, sm_block_cnt * blockDim.x);
          sm_syncer->sync(sm_block_cnt);
        }
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
  for (int i = tid; i < nrecv_proxy; i += blockDim.x) recv_proxy_channels[i].flush();
  for (int i = tid; i < nsend_proxy; i += blockDim.x) send_proxy_channels[i].flush();
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

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
           int* sm_block_idx_block, int* sm_block_cnt_block, mscclpp::DeviceSyncer** sm_syncer_block,
           const uint64_t* data_start, const uint64_t nelem_per_send, const uint64_t* nelem_total) {
  const int bid = blockIdx.x;
  const int sm_block_idx = sm_block_idx_block[bid];
  const int sm_block_cnt = sm_block_cnt_block[bid];
  mscclpp::DeviceSyncer* sm_syncer = sm_syncer_block[bid];

  if (sm_block_idx == 0 && sm_block_cnt > 1) new(sm_syncer) mscclpp::DeviceSyncer();
  deviceSyncer.sync(gridDim.x);

  if (sm_block_idx == 0) {
    threadblockCall(recv_sm_channels == nullptr ? nullptr : &recv_sm_channels[block_recv_sm_ch_starts[bid]], 
                    send_sm_channels == nullptr ? nullptr : &send_sm_channels[block_send_sm_ch_starts[bid]],
                    recv_proxy_channels == nullptr ? nullptr : &recv_proxy_channels[block_recv_proxy_ch_starts[bid]], 
                    send_proxy_channels == nullptr ? nullptr : &send_proxy_channels[block_send_proxy_ch_starts[bid]],
                    &recv_scratches[block_scratch_starts[bid]], nrecvs_sm[bid], nsends_sm[bid], nrecvs_proxy[bid], nsends_proxy[bid],
                    node_types[bid], scratch_size, data,
                    sm_block_cnt, sm_syncer,
                    data_start[bid], nelem_per_send, nelem_total[bid]);
  } else {
    assert(recv_sm_channels != nullptr);
    assert(nrecvs_sm[bid] == 1);
    parallel_sm_threadblockCall(&recv_sm_channels[block_recv_sm_ch_starts[bid]], 
                                sm_block_idx, sm_block_cnt, sm_syncer,
                                data_start[bid], nelem_per_send, nelem_total[bid]);
  }
}