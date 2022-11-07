/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/attention.h"
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
//#include <sys/time.h>
#include <cmath>
using namespace std;

#include <mma.h>
using namespace nvcuda;

namespace fastertransformerv3 {

#define SKEW_HALF 8 // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head>
__global__
    // __launch_bounds__(512,4)//THREADS_PER_BLOCK
    void
    wmma_attention_kernel(const half2 *query, const half2 *query_bias,
                          const half2 *key, const half2 *key_bias,
                          const half2 *value, const half2 *value_bias,
                          const __half *attention_mask,
                          __half *attention_output, const int batch_size,
                          const int head_num, const int seq_len,
                          const half2 scaler) {
  const int half_size_per_head = size_per_head / 2;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = (threadIdx.x & 0x1f);

  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  const int half_hidden_dim = head_num * half_size_per_head;
  const int head_offset = head_id * half_size_per_head;
  const int bias_id = head_offset + warp_tid;

  // loading Query & Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos =
        (bid * seq_len + seq_id) * half_hidden_dim + head_offset + warp_tid;
    half2 tmp = __hadd2(__ldg(&query[pos]), __ldg(&query_bias[bias_id]));
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hmul2(tmp, scaler);

    *(__half2 *)(*s_kv + offset) =
        __hadd2(__ldg(&key[pos]), __ldg(&key_bias[bias_id]));
  }

  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);

      float mask =
          to_id[i] < seq_len
              ? (float)__ldg(&attention_mask[bid * seq_len * seq_len +
                                             from_id * seq_len + to_id[i]])
              : 0.0f;
      mask = (1.0f - mask) * (-10000.0f);

      logits[i] = to_id[i] < seq_len
                      ? (float)(s_logits[from_id][to_id[i]]) + mask
                      : -1e20f;
      max_val = max(max_val, logits[i]);
    }

    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += (to_id[i] < seq_len) ? logits[i] : 0.0f;
    }

    sum_val = warpReduceSum(sum_val);

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] =
            (__half)(to_id[i] < seq_len ? (logits[i] / (sum_val + 1e-6f))
                                        : 0.0f);
  }

  // loading Value
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos =
        (bid * seq_len + seq_id) * half_hidden_dim + head_offset + warp_tid;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] =
        __hadd2(__ldg(&value[pos]), __ldg(&value_bias[bias_id]));
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;

  __syncthreads();

  //* V
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat,
                             s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos =
        (bid * seq_len + from_id) * half_hidden_dim + head_offset + warp_tid;
    ((__half2 *)(attention_output))[pos] =
        ((__half2 *)(s_query[from_id]))[warp_tid];
  }
}

template <const int max_seq_len, const int size_per_head>
__global__
    // __launch_bounds__(256)//THREADS_PER_BLOCK
    void
    wmma_attention_kernel_LE32(const half2 *query, const half2 *query_bias,
                               const half2 *key, const half2 *key_bias,
                               const half2 *value, const half2 *value_bias,
                               const __half *attention_mask,
                               __half *attention_output, const int batch_size,
                               const int head_num, const int seq_len,
                               const half2 scaler) {
  const int half_size_per_head = size_per_head / 2;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];
  __shared__ __half s_value[max_seq_len][size_per_head + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = (threadIdx.x & 0x1f);

  const int bid = blockIdx.x / head_num;
  const int head_id = blockIdx.x % head_num;

  const int half_hidden_dim = head_num * half_size_per_head;
  const int head_offset = head_id * half_size_per_head;
  const int bias_id = head_offset + warp_tid;

  // loading Query & Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos =
        (bid * seq_len + seq_id) * half_hidden_dim + head_offset + warp_tid;
    half2 tmp = __hadd2(__ldg(&query[pos]), __ldg(&query_bias[bias_id]));
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hmul2(tmp, scaler);
    *(__half2 *)(*s_kv + offset) =
        __hadd2(__ldg(&key[pos]), __ldg(&key_bias[bias_id]));
    *(__half2 *)(*s_value + offset) =
        __hadd2(__ldg(&value[pos]), __ldg(&value_bias[bias_id]));
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_value[seq_id]))[warp_tid] = 0.0f;

  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);

    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    float max_val = -1e20f;
    int to_id = warp_tid;

    float mask = to_id < seq_len
                     ? (float)__ldg(&attention_mask[bid * seq_len * seq_len +
                                                    from_id * seq_len + to_id])
                     : 0.0f;
    mask = (1.0f - mask) * (-10000.0f);
    float logits =
        to_id < seq_len ? (float)(s_logits[from_id][to_id]) + mask : max_val;
    max_val = warpReduceMax(logits);

    logits = __expf(logits - max_val);
    float sum_val = (to_id < seq_len) ? logits : 0.0f;
    sum_val = warpReduceSum(sum_val);

    if (to_id < max_seq_len)
      s_logits[from_id][to_id] =
          to_id < seq_len ? logits / (sum_val + 1e-6f) : 0.0f;
  }

  __syncthreads();

  //* V
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_N;
#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat,
                             s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_value[k * WMMA_K] + warp_to_offset,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos =
        (bid * seq_len + from_id) * half_hidden_dim + head_offset + warp_tid;
    ((__half2 *)(attention_output))[pos] =
        ((__half2 *)(s_query[from_id]))[warp_tid];
  }
}

template <OperationType OpType_>
void Attention<OpType_>::fused_infer(
    const DataType_ *query, const DataType_ *key, const DataType_ *value,
    const DataType_ *atten_mask, DataType_ *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream) {
  dim3 grid(batch_size * head_num_);
  dim3 block;

  if (OpType_ == OperationType::HALF) {
    const half2 *query_ptr = (const half2 *)query;
    const half2 *query_bias_ptr = (const half2 *)param_.attr_bias_Q;
    const half2 *key_ptr = (const half2 *)key;
    const half2 *key_bias_ptr = (const half2 *)param_.attr_bias_K;
    const half2 *value_ptr = (const half2 *)value;
    const half2 *value_bias_ptr = (const half2 *)param_.attr_bias_V;
    half2 scaler;
    scaler.x = (__half)(0.125f), scaler.y = (__half)(0.125f);

    if (seq_len <= 16) {
      block.x = 128;
      wmma_attention_kernel_LE32<16, 64><<<grid, block, 0, stream>>>(
          query_ptr, query_bias_ptr, key_ptr, key_bias_ptr, value_ptr,
          value_bias_ptr, (__half *)atten_mask, (__half *)attention_output,
          batch_size, head_num_, seq_len, scaler);
    } else if (seq_len <= 32) {
      block.x = 256;
      wmma_attention_kernel_LE32<32, 64><<<grid, block, 0, stream>>>(
          query_ptr, query_bias_ptr, key_ptr, key_bias_ptr, value_ptr,
          value_bias_ptr, (__half *)atten_mask, (__half *)attention_output,
          batch_size, head_num_, seq_len, scaler);
    } else if (seq_len <= 48) {
      block.x = 384;
      wmma_attention_kernel<48, 64><<<grid, block, 0, stream>>>(
          query_ptr, query_bias_ptr, key_ptr, key_bias_ptr, value_ptr,
          value_bias_ptr, (__half *)atten_mask, (__half *)attention_output,
          batch_size, head_num_, seq_len, scaler);
    } else if (seq_len <= 64) {
      block.x = 512;
      wmma_attention_kernel<64, 64><<<grid, block, 0, stream>>>(
          query_ptr, query_bias_ptr, key_ptr, key_bias_ptr, value_ptr,
          value_bias_ptr, (__half *)atten_mask, (__half *)attention_output,
          batch_size, head_num_, seq_len, scaler);
    }
  }
}

template void Attention<OperationType::FP32>::fused_infer(
    const float *query, const float *key, const float *value,
    const float *atten_mask, float *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream);

template void Attention<OperationType::HALF>::fused_infer(
    const __half *query, const __half *key, const __half *value,
    const __half *atten_mask, __half *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream);
} // namespace fastertransformerv3
