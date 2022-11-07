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
//     cudaFuncSetAttribute(wmma_attention_long_backward_kernel<256, 64>,
//     cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024)
// }

template <const int max_seq_len, const int size_per_head>
__global__ void wmma_attention_backward_kernel(
    const __half *grad_out, const __half *softmax_output, const half2 *q,
    const half2 *k, const half2 *v, half2 *grad_q, half2 *grad_k, half2 *grad_v,
    const int seq_len, const half2 scaler, const float scale,
    const uint8_t *dropout_mask, const __half *softmax_dropout_output) {
  extern __shared__ __half base[];
  __half(*s_softmax)[max_seq_len + SKEW_HALF] =
      (__half(*)[max_seq_len + SKEW_HALF]) base;
  __half(*s_grad)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(
          base + max_seq_len * (max_seq_len + SKEW_HALF));
  __half(*s_qkv)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(
          base +
          max_seq_len * (max_seq_len + SKEW_HALF + size_per_head + SKEW_HALF));
  __half(*s_softmax_out)[max_seq_len + SKEW_HALF] =
      (__half(*)[max_seq_len + SKEW_HALF])(
          base + max_seq_len * (max_seq_len + SKEW_HALF));

  const int warpNums = (blockDim.x >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = (threadIdx.x & 0x1f);
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = blockIdx.y * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading grad_out & s_softmax
  const __half *tmp_ptr =
      scale > 1.0f ? softmax_dropout_output : softmax_output;
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    int offset = from_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_grad + offset) = __ldg(&((const __half2 *)grad_out)[pos]);

    for (int to_id = warp_tid; to_id < seq_len; to_id += 32) {
      int softmax_offset =
          ((blockIdx.y * gridDim.x + blockIdx.x) * seq_len + from_id) *
              seq_len +
          to_id;
      s_softmax[from_id][to_id] = __ldg(&tmp_ptr[softmax_offset]);
    }
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len;
       seq_id += warpNums) {
    ((float *)(s_grad[seq_id]))[warp_tid] = 0.0f;
    for (int to_id = warp_tid; to_id < (seq_len + 1) / 2; to_id += 32)
      ((float *)(s_softmax[seq_id]))[to_id] = 0.0f;
  }

  __syncthreads();

  // compute s_qkv = s_softmax.T * s_grad
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        softmax_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        gradout_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> gradV_mat;
    wmma::fill_fragment(gradV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(softmax_mat,
                             s_softmax[k * WMMA_K] + warp_from_offset,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(gradout_mat, s_grad[k * WMMA_K] + warp_to_offset,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(gradV_mat, softmax_mat, gradout_mat, gradV_mat);
    }
    wmma::store_matrix_sync(s_qkv[warp_from_offset] + warp_to_offset, gradV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }

  __syncthreads();

  // saving s_qkv -> grad_V & loading Value -> s_qkv
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    grad_v[pos] = ((__half2 *)(s_qkv[from_id]))[warp_tid];
    ((__half2 *)(s_qkv[from_id]))[warp_tid] = __ldg(&v[pos]);
  }

  __syncthreads();

  // compute grad_softmax_out = s_grad * v.T
  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        gradout_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        gradsoftmax_mat;
    wmma::fill_fragment(gradsoftmax_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(gradout_mat, s_grad[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_qkv[warp_to_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(gradsoftmax_mat, gradout_mat, V_mat, gradsoftmax_mat);
    }
    wmma::store_matrix_sync(s_softmax[warp_from_offset] + warp_to_offset,
                            gradsoftmax_mat, max_seq_len + SKEW_HALF,
                            wmma::mem_row_major);
  }

  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    float sum = 0.0f;
    for (int to_id = warp_tid; to_id < seq_len; to_id += 32) {
      int pos_id = ((blockIdx.y * gridDim.x + blockIdx.x) * seq_len + from_id) *
                       seq_len +
                   to_id;
      float softmax_out = (float)__ldg(&softmax_output[pos_id]);
      s_softmax_out[from_id][to_id] = softmax_out;

      float grad_softmax = (float)s_softmax[from_id][to_id];
      // compute dropout grad
      if (scale > 1.0f) {
        grad_softmax = dropout_bw(grad_softmax, scale, pos_id, dropout_mask);
        s_softmax[from_id][to_id] = (__half)grad_softmax;
      }

      sum += softmax_out * grad_softmax;
    }

    // compute softmax grad & scale
    sum = warpReduceSum(sum);
    for (int to_id = warp_tid; to_id < seq_len; to_id += 32)
      s_softmax[from_id][to_id] =
          (__half)(((float)s_softmax[from_id][to_id] - sum) *
                   (float)s_softmax_out[from_id][to_id] * (float)scaler.x);
  }

  // K dim clear 0
  // for(int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id +=
  // warpNums)
  //     ((float *)(s_qkv[seq_id]))[warp_tid] = 0.0f;

  __syncthreads();

  // loading Query
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * half_hidden_dim + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_qkv + offset) = __ldg(&q[pos]);
  }

  // K dim clear 0
  // for(int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id +=
  // warpNums)
  //     ((float *)(s_qkv[seq_id]))[warp_tid] = 0.0f;

  __syncthreads();

  // compute grad_K = s_grad.T * q
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::col_major>
        gradout_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        Q_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> gradK_mat;
    wmma::fill_fragment(gradK_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < from_size; k++) {
      wmma::load_matrix_sync(gradout_mat,
                             s_softmax[k * WMMA_K] + warp_from_offset,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(Q_mat, s_qkv[k * WMMA_K] + warp_to_offset,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(gradK_mat, gradout_mat, Q_mat, gradK_mat);
    }
    wmma::store_matrix_sync(s_grad[warp_from_offset] + warp_to_offset,
                            gradK_mat, size_per_head + SKEW_HALF,
                            wmma::mem_row_major);
  }

  __syncthreads();

  // saving -> grad_k and loading Key
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    grad_k[pos] = ((__half2 *)(s_grad[from_id]))[warp_tid];
    ((__half2 *)(s_qkv[from_id]))[warp_tid] = __ldg(&k[pos]);
  }

  // K dim clear 0
  // for(int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id +=
  // warpNums)
  //     ((float *)(s_qkv[seq_id]))[warp_tid] = 0.0f;

  __syncthreads();

  // compute grad_Q = s_grad * k
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        gradout_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> gradQ_mat;
    wmma::fill_fragment(gradQ_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(gradout_mat,
                             s_softmax[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_qkv[k * WMMA_K] + warp_to_offset,
                             size_per_head + SKEW_HALF);
      wmma::mma_sync(gradQ_mat, gradout_mat, K_mat, gradQ_mat);
    }
    wmma::store_matrix_sync(s_grad[warp_from_offset] + warp_to_offset,
                            gradQ_mat, size_per_head + SKEW_HALF,
                            wmma::mem_row_major);
  }

  __syncthreads();

  // saving grad_Q -> grad_q
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    grad_q[pos] = ((__half2 *)(s_grad[from_id]))[warp_tid];
  }
}

template <const int max_seq_len, const int size_per_head>
__global__ void wmma_attention_long_backward_kernel(
    const __half *grad_out, const __half *softmax_output, const half2 *q,
    const half2 *k, const half2 *v, half2 *grad_q, half2 *grad_k, half2 *grad_v,
    const int seq_len, const half2 scaler, const float scale,
    const uint8_t *dropout_mask, __half *softmax_dropout_output) {
  // const int split_seq_len = 64;

  // extern __shared__ __half base[];
  // __half (*s_qkv)[size_per_head    + SKEW_HALF] = (__half
  // (*)[size_per_head + SKEW_HALF])base;
  // __half (*s_qkv)[size_per_head + SKEW_HALF] = (__half (*)[size_per_head
  // + SKEW_HALF])(base + (seq_len) * (size_per_head + SKEW_HALF));
  // __half (*s_logits)[max_seq_len  + SKEW_HALF] = (__half (*)[max_seq_len
  // + SKEW_HALF])(base + (split_seq_len + seq_len) * (size_per_head +
  // SKEW_HALF));

  // const int warpNums = (blockDim.x  >> 5);
  // const int warpId   = (threadIdx.x >> 5);
  // const int warp_tid = (threadIdx.x & 0x1f);
  // const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  // const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;
  // const int batch_seq_offset = blockIdx.z * seq_len;

  // load grad_out & transpose
  // for(int from_id = warpId; from_id < split_seq_len; from_id += warpNums)
  // {
  //     int pos = (batch_seq_offset + blockIdx.y * split_seq_len + from_id) *
  //     half_hidden_dim + thread_offset;
  //     ((__half2 *)(attention_output))[pos] = ((__half2
  //     *)(s_qkv[from_id]))[warp_tid];
  // }

  __syncthreads();
}

#define WMMA_ATTENTION_BACKWARD(SEQ_LEN, SIZE_PER_HEAD)                        \
  wmma_attention_backward_kernel<SEQ_LEN, SIZE_PER_HEAD>                       \
      <<<grid, block, shared_memory_size, param.stream>>>(                     \
          (__half *)param.grad_out, (__half *)param.softmax_output, q_ptr,     \
          k_ptr, v_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr, param.seq_len,     \
          scaler, dropout_scale, param.dropout_mask,                           \
          (__half *)param.softmax_dropout_output)

template <OperationType OpType>
void FuseAttention<OpType>::backward(FuseAttentionBackwardParam param) {
  if (OpType == OperationType::HALF) {
    const half2 *q_ptr = (const half2 *)param.input_q;
    const half2 *k_ptr = (const half2 *)param.input_k;
    const half2 *v_ptr = (const half2 *)param.input_v;
    half2 *grad_q_ptr = (half2 *)param.grad_q;
    half2 *grad_k_ptr = (half2 *)param.grad_k;
    half2 *grad_v_ptr = (half2 *)param.grad_v;

    float scale = 1.0f / sqrtf(param.size_per_head);
    half2 scaler(scale, scale);

    float dropout_scale = 1.0f / (1.0f - param.dropout_rate);
    if (param.seq_len == 128) {
      // todo: set one time
      cudaFuncSetAttribute(wmma_attention_long_backward_kernel<128, 64>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           64 * 1024);

      const int split_k = 128 / 64;
      dim3 grid(param.head_num, split_k, param.batch_size), block;
      block.x = 32 * (4 * 8);
      wmma_attention_long_backward_kernel<128, 64>
          <<<grid, block, 64 * 1024, param.stream>>>(
              (__half *)param.grad_out, (__half *)param.softmax_output, q_ptr,
              k_ptr, v_ptr, grad_q_ptr, grad_k_ptr, grad_v_ptr, param.seq_len,
              scaler, dropout_scale, param.dropout_mask,
              (__half *)param.softmax_dropout_output);
    } else if (param.seq_len <= 80) {
      int max_seq_len = (param.seq_len + 15) / 16 * 16;
      int softmax_size = max_seq_len * (max_seq_len + SKEW_HALF);
      int shared_memory_size =
          softmax_size +
          max(softmax_size,
              2 * (max_seq_len * (param.size_per_head + SKEW_HALF)));
      shared_memory_size *= sizeof(DataType_);

      dim3 grid(param.head_num, param.batch_size), block;
      block.x = 32 * ((param.seq_len + 15) / 16) *
                max(((param.seq_len + 15) / 16), 64 / 16);
      if (param.seq_len <= 16)
        WMMA_ATTENTION_BACKWARD(16, 64);
      else if (param.seq_len <= 32)
        WMMA_ATTENTION_BACKWARD(32, 64);
      else if (param.seq_len <= 48)
        WMMA_ATTENTION_BACKWARD(48, 64);
      else if (param.seq_len <= 64)
        WMMA_ATTENTION_BACKWARD(64, 64);
      else if (param.seq_len <= 80)
        WMMA_ATTENTION_BACKWARD(80, 64);
    }
  } else {
    printf("FP32 fuse_attention backward op is not supported\n");
  }
}

template void
FuseAttention<OperationType::FP32>::backward(FuseAttentionBackwardParam param);
template void
FuseAttention<OperationType::HALF>::backward(FuseAttentionBackwardParam param);
} // namespace fastertransformerv4