/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include <cuda_fp16.h>

namespace fastertransformerv3 {
// ***************************** GEMM ****************************************

void dense_layer_kernel_launcher(const float *in, const float *weight,
                                 float *out, const int M, const int K,
                                 const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = -1);

void dense_layer_kernel_launcher(const __half *in, const __half *weight,
                                 __half *out, const int M, const int K,
                                 const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);

// ***************************** add_bias + gelu *****************************

template <typename T> __inline__ __device__ T gelu(T val) {
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f *
                                    (val + 0.044715f * val * val * val))));
  return val * cdf;
}

template <> __inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);
  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

__global__ void add_bias_gelu(float *output, const float *bias, const int M,
                              const int N);

__global__ void add_bias_gelu(__half *output, const __half *bias, const int M,
                              const int N);

template <typename T> __inline__ __device__ T swish(T val) {
  return val / (1.0f + __expf(-(float)val));
}

template <> __inline__ __device__ half2 swish(half2 val) {
  float2 tmp = __half22float2(val);
  tmp.x = tmp.x / (1.0f + __expf(-tmp.x));
  tmp.y = tmp.y / (1.0f + __expf(-tmp.y));
  return __float22half2_rn(tmp);
}

__global__ void add_bias_swish(float *output, const float *bias, const int M,
                               const int N);

__global__ void add_bias_swish(__half *output, const __half *bias, const int M,
                               const int N);

template <ActType act, typename T>
__global__ void add_bias_act(T *output, const T *bias, const int M,
                             const int N) {
  int row_offset = blockIdx.x * N;
  for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    T out = output[row_offset + tid] + __ldg(&bias[tid]);
    out = act_fun<act>(out);
    output[row_offset + tid] = out;
  }
}

// ************************** build_sequence_length_padding_offset
// **************************

template <typename T> __inline__ __device__ T warpPrefixSum(int id, T count) {
  for (int i = 1; i < 32; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

template <typename T>
__global__ void parallel_prefix(const T *atten_mask, int *batch_idx,
                                int *word_idx, const int batch_size,
                                const int max_seq_len) {
  const int tid = threadIdx.x;
  const int warp_count = blockDim.x >> 5;
  int warp_id = tid >> 5;
  int warp_tid = tid & 0x1F;

  extern __shared__ int base[];

  int *seq_len = base;
  int *seq_offset = base + batch_size;

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < max_seq_len ? atten_mask[wid * max_seq_len * max_seq_len + i]
                               : (T)0.0f;
      count += __popc(__ballot_sync(0xFFFFFFFF, mask >= (T)0.5f));
    }
    if (warp_tid == 0)
      seq_len[wid] = count;
  }

  __syncthreads();

  if (warp_id == 0) {
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? seq_len[i] : 0;
      temp = warpPrefixSum(warp_tid, offset + len);
      if (i < batch_size)
        seq_offset[i] = temp - len;

      temp = __shfl_sync(0xffffffff, temp, 31);
    }
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  __syncthreads();

  for (int i = tid; i <= batch_size; i += blockDim.x)
    batch_idx[i] = seq_offset[i];

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    for (int i = warp_tid; i < seq_len[wid]; i += 32)
      word_idx[offset + i] = wid * max_seq_len + i;
  }
}

template <typename T>
void build_sequence_length_padding_offset_kernelLauncher(
    const T *atten_mask, int *batch_idx, int *word_idx, int *valid_word_num,
    const int batch_size, const int max_seq_len, cudaStream_t stream) {
  dim3 block(batch_size * 32); // one warp per sequence
  if (block.x > 1024)
    block.x = 1024;
  parallel_prefix<<<1, block, (2 * batch_size + 1) * sizeof(int), stream>>>(
      atten_mask, batch_idx, word_idx, batch_size, max_seq_len);
  cudaMemcpyAsync(valid_word_num, batch_idx + batch_size, sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
}

// *********************** compresse transformer input ***********************

template <typename T>
__global__ void compress_bert_input(const T *from_tensor, T *to_tensor,
                                    const int *batch_idx, const int *word_idx,
                                    int hidden_dim) {
  int offset = __ldg(&word_idx[blockIdx.x]);
  int dst_idx = blockIdx.x * hidden_dim + threadIdx.x;
  int src_idx = offset * hidden_dim + threadIdx.x;
  ((float4 *)to_tensor)[dst_idx] = ((const float4 *)from_tensor)[src_idx];
}

template <typename T>
void compressBertInput_kernelLauncher(const T *from_tensor, T *to_tensor,
                                      int *batch_idx, int *word_idx,
                                      int valid_word_num, int batch_size,
                                      int hidden_dim, cudaStream_t stream) {
  dim3 grid(valid_word_num);
  dim3 block(hidden_dim / 4); // assert(hidden_dim / 4 <= 1024);
  compress_bert_input<<<grid, block, 0, stream>>>(
      from_tensor, to_tensor, batch_idx, word_idx, hidden_dim / 4);
}

// *********************** add bias input ***********************

__global__ void add_bias_input(float *out, const float *input,
                               const float *bias, int m, int n);

__global__ void add_bias_input(__half *out, const __half *input,
                               const __half *bias, int m, int n);

__global__ void add_bias_half_input(float *out, const float *input,
                                    const float *bias, int m, int n);

__global__ void add_bias_half_input(__half *out, const __half *input,
                                    const __half *bias, int m, int n);

// *********************** add bias input restore transformer output
// ***********************

__global__ void add_bias_input_restore_output(const float *out,
                                              const float *input,
                                              const float *bias, int m, int n,
                                              float *out2, const int *batch_idx,
                                              const int *word_idx);

__global__ void
add_bias_input_restore_output(const __half *out, const __half *input,
                              const __half *bias, int m, int n, __half *out2,
                              const int *batch_idx, const int *word_idx);

__global__ void
add_bias_half_input_restore_output(const float *out, const float *input,
                                   const float *bias, int m, int n, float *out2,
                                   const int *batch_idx, const int *word_idx);

__global__ void add_bias_half_input_restore_output(
    const __half *out, const __half *input, const __half *bias, int m, int n,
    __half *out2, const int *batch_idx, const int *word_idx);

} // namespace fastertransformerv3