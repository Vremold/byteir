/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/remove_padding.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

namespace fastertransformerv4 {
template <typename T> __inline__ __device__ T warpPrefixSum(int id, T count) {
  for (int i = 1; i < 32; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

template <typename T>
__global__ void parallel_prefix(const T *atten_mask, int *word_idx,
                                const int batch_size, const int seq_len) {
  const int tid = threadIdx.x;
  const int warp_count = blockDim.x >> 5;
  int warp_id = tid >> 5;
  int warp_tid = tid & 0x1F;

  extern __shared__ int base[];
  int *valid_seq_len = base;
  int *seq_offset = base + batch_size;

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    for (int i = warp_tid; i < (seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < seq_len ? atten_mask[wid * seq_len * seq_len + i] : (T)0.0f;
      count += __popc(__ballot_sync(0xFFFFFFFF, mask > (T)0.5f));
    }
    if (warp_tid == 0)
      valid_seq_len[wid] = count;
  }

  __syncthreads();

  if (warp_id == 0) {
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? valid_seq_len[i] : 0;
      temp = warpPrefixSum(warp_tid, offset + len);
      if (i < batch_size)
        seq_offset[i] = temp - len;

      temp = __shfl_sync(0xffffffff, temp, 31);
    }
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  __syncthreads();

  const unsigned int t_mask = (1 << warp_tid) - 1;
  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    // for(int i = warp_tid; i < valid_seq_len[wid]; i += 32)
    //     word_idx[offset + i] = wid * seq_len + i;
    for (int i = warp_tid; i < (seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < seq_len ? __ldg(&atten_mask[wid * seq_len * seq_len + i])
                           : (T)0.0f;
      uint32_t active_mask = __ballot_sync(0xFFFFFFFF, mask > (T)0.5f);
      int seq_pos = __popc(active_mask & t_mask);
      if (mask > (T)0.5f)
        word_idx[offset + seq_pos] = wid * seq_len + i;
      offset += __popc(active_mask);
    }
  }

  // for(int i = tid; i <= batch_size; i += blockDim.x)
  //     batch_idx[i] = seq_offset[i];
  if (tid == 0)
    word_idx[batch_size * seq_len] = seq_offset[batch_size];
}

template <typename T>
__global__ void compress_input(const T *from_tensor, T *to_tensor,
                               const int *word_idx, int hidden_dim,
                               int valid_word_num) {
  int dst_idx = blockIdx.x * hidden_dim + threadIdx.x;
  if (blockIdx.x < valid_word_num) {
    int src_idx = __ldg(&word_idx[blockIdx.x]) * hidden_dim + threadIdx.x;
    ((float4 *)to_tensor)[dst_idx] = ((const float4 *)from_tensor)[src_idx];
  } else
    ((float4 *)to_tensor)[dst_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template <typename T>
__global__ void restore_input(const T *from_tensor, T *to_tensor,
                              const int *word_idx, int hidden_dim) {
  int src_idx = blockIdx.x * hidden_dim + threadIdx.x;
  int dst_idx = __ldg(&word_idx[blockIdx.x]) * hidden_dim + threadIdx.x;
  ((float4 *)to_tensor)[dst_idx] = ((const float4 *)from_tensor)[src_idx];
}

template <OperationType OpType>
void RemovePadding<OpType>::compress(RemovePaddingParam param) {
  const int hidden_dim =
      (OpType == OperationType::HALF) ? param.hidden_dim / 2 : param.hidden_dim;
  dim3 grid((param.valid_word_num + 7) / 8 * 8);
  dim3 block(hidden_dim / 4);
  compress_input<<<grid, block, 0, param.stream>>>(
      param.input, param.output, param.word_idx, hidden_dim / 4,
      param.valid_word_num);
}

template <OperationType OpType>
void RemovePadding<OpType>::restore(RemovePaddingParam param) {
  cudaMemsetAsync(param.output, 0,
                  param.batch_size * param.seq_len * param.hidden_dim *
                      sizeof(DataType_),
                  param.stream);

  const int hidden_dim =
      (OpType == OperationType::HALF) ? param.hidden_dim / 2 : param.hidden_dim;
  dim3 grid(param.valid_word_num);
  dim3 block(hidden_dim / 4);
  restore_input<<<grid, block, 0, param.stream>>>(
      param.input, param.output, param.word_idx, hidden_dim / 4);
}

template <OperationType OpType>
void RemovePadding<OpType>::get_valid_word_index(GetValidWordIndexParam param) {
  dim3 block(std::min(param.batch_size * 32, 1024)); // one warp per sequence
  parallel_prefix<<<1, block, (2 * param.batch_size + 1) * sizeof(int),
                    param.stream>>>(param.attention_mask, param.word_idx,
                                    param.batch_size, param.seq_len);
  cudaMemcpyAsync(param.h_valid_word_num_ptr,
                  param.word_idx + param.batch_size * param.seq_len,
                  sizeof(int), cudaMemcpyDeviceToHost, param.stream);
}

template void
RemovePadding<OperationType::FP32>::compress(RemovePaddingParam param);
template void
RemovePadding<OperationType::HALF>::compress(RemovePaddingParam param);

template void
RemovePadding<OperationType::FP32>::restore(RemovePaddingParam param);
template void
RemovePadding<OperationType::HALF>::restore(RemovePaddingParam param);

template void RemovePadding<OperationType::FP32>::get_valid_word_index(
    GetValidWordIndexParam param);
template void RemovePadding<OperationType::HALF>::get_valid_word_index(
    GetValidWordIndexParam param);
} // namespace fastertransformerv4