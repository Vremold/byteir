/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v4/includes/fuse_attention.h"
#include "fastertransformer_v4/includes/reduce.h"
#include "fastertransformer_v4/includes/utils.h"
using namespace std;

#include <mma.h>
using namespace nvcuda;

namespace fastertransformerv4 {
#define SKEW_HALF 8 // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// template<OperationType OpType>
// void Attention<OpType>::set_shared_memory()
// {
//     cudaFuncSetAttribute(wmma_attention_long_forward_kernel<256, 64>,
//     cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024)
// }

template <const int max_seq_len, const int size_per_head>
__global__ void wmma_attention_forward_kernel(
    const half2 *q, const half2 *k, const half2 *v,
    const __half *attention_mask, __half *softmax_output,
    __half *attention_output, const int seq_len, const half2 scaler,
    const float dropout_rate, const int seed, uint8_t *dropout_mask,
    __half *softmax_dropout_output) {
  __shared__ __half s_kv[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_query[max_seq_len][size_per_head + SKEW_HALF];
  __shared__ __half s_logits[max_seq_len][max_seq_len + SKEW_HALF];

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = (threadIdx.x & 0x1f);
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = blockIdx.y * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query & Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * half_hidden_dim + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hmul2(__ldg(&q[pos]), scaler);
    *(__half2 *)(*s_kv + offset) = __ldg(&k[pos]);
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
              ? (float)__ldg(
                    &attention_mask[(batch_seq_offset + from_id) * seq_len +
                                    to_id[i]])
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

    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len) {
        __half softmax_result =
            (__half)(to_id[i] < seq_len ? __fdividef(logits[i], sum_val)
                                        : 0.0f);
        if (to_id[i] < seq_len) {
          int offset =
              ((blockIdx.y * gridDim.x + blockIdx.x) * seq_len + from_id) *
                  seq_len +
              to_id[i];
          softmax_output[offset] = softmax_result;
          if (dropout_rate > 0.0f) {
            softmax_result =
                (__half)dropout_fw((float)softmax_result, dropout_rate, seed,
                                   offset, dropout_mask);
            softmax_dropout_output[offset] = softmax_result;
          }
        }
        s_logits[from_id][to_id[i]] = softmax_result;
      }
  }

  // loading Value
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] = __ldg(&v[pos]);
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
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] =
        ((__half2 *)(s_query[from_id]))[warp_tid];
  }
}

template <const int max_seq_len, const int size_per_head>
__global__ void wmma_attention_long_forward_kernel(
    const half2 *q, const half2 *k, const half2 *v,
    const __half *attention_mask, __half *softmax_output,
    __half *attention_output, const int seq_len, const half2 scaler,
    const float dropout_rate, const int seed, uint8_t *dropout_mask,
    __half *softmax_dropout_output) {
  const int split_seq_len = 64;

  extern __shared__ __half base[];
  __half(*s_kv)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF]) base;
  __half(*s_query)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(
          base + (seq_len) * (size_per_head + SKEW_HALF));
  __half(*s_logits)[max_seq_len + SKEW_HALF] =
      (__half(*)[max_seq_len + SKEW_HALF])(
          base + (split_seq_len + seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = (threadIdx.x & 0x1f);
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = blockIdx.z * seq_len;

  // loading Query
  for (int seq_id = warpId; seq_id < split_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + blockIdx.y * split_seq_len + seq_id) *
                  half_hidden_dim +
              thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hmul2(__ldg(&q[pos]), scaler);
  }

  // loading Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * half_hidden_dim + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_kv + offset) = __ldg(&k[pos]);
  }

  __syncthreads();

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;
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
  for (int from_id = warpId; from_id < split_seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);

      float mask =
          to_id[i] < seq_len
              ? (float)__ldg(
                    &attention_mask[(batch_seq_offset +
                                     blockIdx.y * split_seq_len + from_id) *
                                        seq_len +
                                    to_id[i]])
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

    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len) {
        __half softmax_result =
            (__half)(to_id[i] < seq_len ? __fdividef(logits[i], sum_val)
                                        : 0.0f);
        if (to_id[i] < seq_len) {
          int offset =
              ((blockIdx.y * gridDim.x + blockIdx.x) * seq_len + from_id) *
                  seq_len +
              to_id[i];
          softmax_output[offset] = softmax_result;
          if (dropout_rate > 0.0f) {
            softmax_result =
                (__half)dropout_fw((float)softmax_result, dropout_rate, seed,
                                   offset, dropout_mask);
            softmax_dropout_output[offset] = softmax_result;
          }
        }
        s_logits[from_id][to_id[i]] = softmax_result;
      }
  }

  // loading Value
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] = __ldg(&v[pos]);
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

  for (int from_id = warpId; from_id < split_seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + blockIdx.y * split_seq_len + from_id) *
                  half_hidden_dim +
              thread_offset;
    ((__half2 *)(attention_output))[pos] =
        ((__half2 *)(s_query[from_id]))[warp_tid];
  }
}

#define WMMA_ATTENTION_FORWARD(SEQ_LEN, SIZE_PER_HEAD)                         \
  wmma_attention_forward_kernel<SEQ_LEN, SIZE_PER_HEAD>                        \
      <<<grid, block, 0, param.stream>>>(                                      \
          q_ptr, k_ptr, v_ptr, (__half *)param.mask,                           \
          (__half *)param.softmax_output, (__half *)param.attention_output,    \
          param.seq_len, scaler, param.dropout_rate, seed, param.dropout_mask, \
          (__half *)param.softmax_dropout_output)

template <OperationType OpType>
void FuseAttention<OpType>::forward(FuseAttentionForwardParam param) {
  if (OpType == OperationType::HALF) {
    const half2 *q_ptr = (const half2 *)param.input_q;
    const half2 *k_ptr = (const half2 *)param.input_k;
    const half2 *v_ptr = (const half2 *)param.input_v;

    const float scale = 1.0f / sqrtf(param.size_per_head);
    const half2 scaler(scale, scale);

    const int seed = param.dropout_rate > 0.0f ? generate_random_seed() : 0;

    if (param.seq_len == 128) {
      // todo: set one time
      cudaFuncSetAttribute(wmma_attention_long_forward_kernel<128, 64>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           64 * 1024);

      const int split_k = 128 / 64;
      dim3 grid(param.head_num, split_k, param.batch_size), block;
      block.x = 32 * (4 * 8);
      wmma_attention_long_forward_kernel<128, 64>
          <<<grid, block, 64 * 1024, param.stream>>>(
              q_ptr, k_ptr, v_ptr, (__half *)param.mask,
              (__half *)param.softmax_output, (__half *)param.attention_output,
              param.seq_len, scaler, param.dropout_rate, seed,
              param.dropout_mask, (__half *)param.softmax_dropout_output);
    } else if (param.seq_len <= 80) {
      dim3 grid(param.head_num, param.batch_size), block;
      block.x = 32 * ((param.seq_len + 15) / 16) *
                max(((param.seq_len + 15) / 16), 64 / 16);
      if (param.seq_len <= 16)
        WMMA_ATTENTION_FORWARD(16, 64);
      else if (param.seq_len <= 32)
        WMMA_ATTENTION_FORWARD(32, 64);
      else if (param.seq_len <= 48)
        WMMA_ATTENTION_FORWARD(48, 64);
      else if (param.seq_len <= 64)
        WMMA_ATTENTION_FORWARD(64, 64);
      else if (param.seq_len <= 80)
        WMMA_ATTENTION_FORWARD(80, 64);
    }
  } else {
    printf("FP32 fuse_attention forward op is not supported\n");
  }
}

template void
FuseAttention<OperationType::FP32>::forward(FuseAttentionForwardParam param);
template void
FuseAttention<OperationType::HALF>::forward(FuseAttentionForwardParam param);
} // namespace fastertransformerv4