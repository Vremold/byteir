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

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4,
                                       int dim_1, int dim_2, int dim_3,
                                       int dim_4) {
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 +
         id4;
}

template <typename T>
__global__ void add_QKV_bias(const T *Q, const T *bias_Q, const T *K,
                             const T *bias_K, const T *V, const T *bias_V,
                             T *q_buf_, T *k_buf_, T *v_buf_,
                             const int batch_size, const int seq_len,
                             const int head_num, const int size_per_head) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = blockIdx.x / seq_len;
  int word_id = blockIdx.x % seq_len;

  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(batch_id, word_id, head_id, id, batch_size,
                               seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  q_buf_[target_id] = Q[tid] + __ldg(&bias_Q[bias_id]);
  k_buf_[target_id] = K[tid] + __ldg(&bias_K[bias_id]);
  v_buf_[target_id] = V[tid] + __ldg(&bias_V[bias_id]);
}

template <>
__global__ void add_QKV_bias(const __half *Q, const __half *bias_Q,
                             const __half *K, const __half *bias_K,
                             const __half *V, const __half *bias_V,
                             __half *q_buf_, __half *k_buf_, __half *v_buf_,
                             const int batch_size, const int seq_len,
                             const int head_num, const int size_per_head);

template <typename T>
__global__ void transpose(const T *src, T *dst, const int batch_size,
                          const int seq_len, const int head_num,
                          const int size_per_head) {
  int head_id = threadIdx.y;
  int tid = threadIdx.x;

  int batch_id = blockIdx.x / seq_len;
  int word_id = blockIdx.x % seq_len;

  int src_offset = batch_id * head_num * seq_len * size_per_head +
                   head_id * seq_len * size_per_head + word_id * size_per_head +
                   tid;
  int dst_offset =
      blockIdx.x * head_num * size_per_head + head_id * size_per_head + tid;

  dst[dst_offset] = src[src_offset];
}

template <>
__global__ void transpose(const __half *src, __half *dst, const int batch_size,
                          const int seq_len, const int head_num,
                          const int size_per_head);

template <typename T>
__global__ void
add_QKV_bias_padding(const T *Q, const T *bias_Q, const T *K, const T *bias_K,
                     const T *V, const T *bias_V, T *q_buf_, T *k_buf_,
                     T *v_buf_, const int batch_size, const int seq_len,
                     const int head_num, const int size_per_head,
                     const int *batch_idx, const int *word_idx) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int offset = word_idx[blockIdx.x];
  int batch_id = offset / seq_len; // batch_idx[blockIdx.x]
  int seq_id = offset % seq_len;   // word_idx[blockIdx.x]

  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(batch_id, seq_id, head_id, id, batch_size,
                               seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  q_buf_[target_id] = Q[tid] + __ldg(&bias_Q[bias_id]);
  k_buf_[target_id] = K[tid] + __ldg(&bias_K[bias_id]);
  v_buf_[target_id] = V[tid] + __ldg(&bias_V[bias_id]);
}

template <>
__global__ void
add_QKV_bias_padding(const __half *Q, const __half *bias_Q, const __half *K,
                     const __half *bias_K, const __half *V,
                     const __half *bias_V, __half *q_buf_, __half *k_buf_,
                     __half *v_buf_, const int batch_size, const int seq_len,
                     const int head_num, const int size_per_head,
                     const int *batch_idx, const int *word_idx);

template <typename T>
__global__ void transpose_rm_padding(const T *src, T *dst, const int batch_size,
                                     const int seq_len, const int head_num,
                                     const int size_per_head,
                                     const int *batch_idx,
                                     const int *word_idx) {
  int head_id = threadIdx.y;
  int tid = threadIdx.x;

  int offset = word_idx[blockIdx.x];
  int batch_id = offset / seq_len; // batch_idx[blockIdx.x]
  int word_id = offset % seq_len;  // word_idx[blockIdx.x]

  int src_offset = batch_id * head_num * seq_len * size_per_head +
                   head_id * seq_len * size_per_head + word_id * size_per_head +
                   tid;
  int dst_offset =
      blockIdx.x * head_num * size_per_head + head_id * size_per_head + tid;

  dst[dst_offset] = src[src_offset];
}

template <>
__global__ void transpose_rm_padding(const __half *src, __half *dst,
                                     const int batch_size, const int seq_len,
                                     const int head_num,
                                     const int size_per_head,
                                     const int *batch_idx, const int *word_idx);
} // namespace fastertransformerv3