/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/fused_multi_head_attention.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include "fastertransformer_v3/includes/utils.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
using namespace std;

// #include <mma.h>
// using namespace nvcuda;

namespace fastertransformerv3 {

template <const int size_per_head>
__global__ void attention_kernel(const float *query, const float *query_bias,
                                 const float *key, const float *key_bias,
                                 const float *value, const float *value_bias,
                                 const float *key_padding_mask,
                                 float *attention_output, const int batch_size,
                                 const int num_heads, const int from_seq_len,
                                 const int to_seq_len, const float scaler) {
  const int max_from_seq_len = 32;
  const int max_to_seq_len = 32;

  __shared__ float s_logits[max_from_seq_len][max_to_seq_len];
  __shared__ float s_query[max_from_seq_len][size_per_head + 1];
  __shared__ float s_kv[max_to_seq_len][size_per_head + 1];

  const int bid = blockIdx.x / num_heads;
  const int head_id = blockIdx.x % num_heads;
  const int input_dim = num_heads * size_per_head;

  int ele_N = from_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / size_per_head;
    int dim_id = tid % size_per_head;
    int offset = head_id * size_per_head + dim_id;
    int pos = seq_id * batch_size * input_dim + bid * input_dim + offset;
    s_query[seq_id][dim_id] = (__ldg(&query[pos]) + __ldg(&query_bias[offset]));
  }

  ele_N = to_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / size_per_head;
    int dim_id = tid % size_per_head;
    int offset = head_id * size_per_head + dim_id;
    int pos = seq_id * batch_size * input_dim + bid * input_dim + offset;
    s_kv[seq_id][dim_id] = __ldg(&key[pos]) + __ldg(&key_bias[offset]);
  }

  __syncthreads();

  ele_N = from_seq_len * to_seq_len;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int from_id = tid / to_seq_len;
    int to_id = tid % to_seq_len;

    float tmp = 0.0f;
    for (int i = 0; i < size_per_head; ++i)
      tmp += s_query[from_id][i] * s_kv[to_id][i];

    float mask =
        (1.0f - __ldg(&key_padding_mask[bid * max_to_seq_len + to_id])) *
        -10000.0f; // query_mask.logical_not()
    s_logits[from_id][to_id] = tmp * scaler + mask;
  }

  __syncthreads();

  // softmax
  for (int from_id = (threadIdx.x >> 5); from_id < from_seq_len;
       from_id += (blockDim.x >> 5)) {
    float max_val = -1e20f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32)
      max_val = max(max_val, s_logits[from_id][to_id]);

    max_val = warpReduceMax(max_val);
    float sum_val = 0.0f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32) {
      float temp = __expf(s_logits[from_id][to_id] - max_val);
      s_logits[from_id][to_id] = temp;
      sum_val += temp;
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32)
      s_logits[from_id][to_id] /= sum_val;
  }

  ele_N = to_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / size_per_head;
    int dim_id = tid % size_per_head;
    int offset = head_id * size_per_head + dim_id;
    int pos = seq_id * batch_size * input_dim + bid * input_dim + offset;
    s_kv[seq_id][dim_id] = __ldg(&value[pos]) + __ldg(&value_bias[offset]);
  }

  __syncthreads();

  ele_N = from_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int from_id = tid / size_per_head;
    int dim_id = tid % size_per_head;

    float tmp = 0.0f;
    for (int i = 0; i < to_seq_len; ++i)
      tmp += s_logits[from_id][i] * s_kv[i][dim_id];

    int pos = from_id * batch_size * input_dim + bid * input_dim +
              head_id * size_per_head + dim_id;
    attention_output[pos] = tmp;
  }
}

template <const int size_per_head>
__global__ void attention_kernel(
    const __half *Q_buf, const __half *query_bias_buf, const __half *K_buf,
    const __half *key_bias_buf, const __half *value, const __half *value_bias,
    const __half *key_padding_mask, __half *attention_output,
    const int batch_size, const int num_heads, const int from_seq_len,
    const int to_seq_len, const __half scaler) {
  const int half_size_per_head = size_per_head / 2;
  const int max_from_seq_len = 32;
  const int max_to_seq_len = 32;

  const half2 *query = (const half2 *)Q_buf;
  const half2 *query_bias = (const half2 *)query_bias_buf;
  const half2 *key = (const half2 *)K_buf;
  const half2 *key_bias = (const half2 *)key_bias_buf;

  __shared__ half2 s_query[max_from_seq_len][half_size_per_head];
  __shared__ half2 s_key[max_to_seq_len][half_size_per_head];
  __shared__ float s_logits[max_from_seq_len][max_to_seq_len];
  __shared__ __half s_value[max_to_seq_len][size_per_head + 1];

  const int bid = blockIdx.x / num_heads;
  const int head_id = blockIdx.x % num_heads;
  const int half_input_dim = num_heads * half_size_per_head;
  const int input_dim = num_heads * size_per_head;

  // loading Query
  int ele_N = from_seq_len * half_size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / half_size_per_head;
    int dim_id = tid % half_size_per_head;
    int offset = head_id * half_size_per_head + dim_id;
    int pos =
        seq_id * batch_size * half_input_dim + bid * half_input_dim + offset;
    s_query[seq_id][dim_id] =
        __hadd2(__ldg(&query[pos]), __ldg(&query_bias[offset]));
  }

  // loading key
  ele_N = to_seq_len * half_size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / half_size_per_head;
    int dim_id = tid % half_size_per_head;
    int offset = head_id * half_size_per_head + dim_id;
    int pos =
        seq_id * batch_size * half_input_dim + bid * half_input_dim + offset;
    s_key[seq_id][dim_id] = __hadd2(__ldg(&key[pos]), __ldg(&key_bias[offset]));
  }

  __syncthreads();

  ele_N = from_seq_len * to_seq_len;
  half2 zero_half = __float2half2_rn(0.0f);
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int from_id = tid / to_seq_len;
    int to_id = tid % to_seq_len;

    half2 tmp = zero_half;
    for (int i = 0; i < half_size_per_head; ++i)
      tmp = __hfma2(s_query[from_id][i], s_key[to_id][i], tmp);

    __half mask = ((__half)1.0f -
                   __ldg(&key_padding_mask[bid * max_to_seq_len + to_id])) *
                  (__half)-10000.0f;
    s_logits[from_id][to_id] = (float)(__hadd(tmp.x, tmp.y) * scaler + mask);
  }

  __syncthreads();

  // softmax
  for (int from_id = (threadIdx.x >> 5); from_id < from_seq_len;
       from_id += (blockDim.x >> 5)) {
    float max_val = -1e20f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32)
      max_val = max(max_val, s_logits[from_id][to_id]);

    max_val = warpReduceMax(max_val);
    float sum_val = 0.0f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32) {
      float temp = __expf(s_logits[from_id][to_id] - max_val);
      s_logits[from_id][to_id] = temp;
      sum_val += temp;
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;
    for (int to_id = (threadIdx.x & 0x1f); to_id < to_seq_len; to_id += 32)
      s_logits[from_id][to_id] /= sum_val;
  }

  // loading V
  ele_N = to_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int seq_id = tid / size_per_head;
    int dim_id = tid % size_per_head;
    int offset = head_id * size_per_head + dim_id;
    int pos = seq_id * batch_size * input_dim + bid * input_dim + offset;
    s_value[seq_id][dim_id] = __ldg(&value[pos]) + __ldg(&value_bias[offset]);
  }

  __syncthreads();

  //* V
  ele_N = from_seq_len * size_per_head;
  for (int tid = threadIdx.x; tid < ele_N; tid += blockDim.x) {
    int from_id = tid / size_per_head;
    int dim_id = tid % size_per_head;

    __half tmp = (__half)0.0f;
    for (int i = 0; i < to_seq_len; ++i)
      tmp += (__half)s_logits[from_id][i] * s_value[i][dim_id];

    int pos = from_id * batch_size * input_dim + bid * input_dim +
              head_id * size_per_head + dim_id;
    attention_output[pos] = tmp;
  }
}

template <OperationType OpType_>
void MultiHeadAttention<OpType_>::fused_infer(
    const DataType_ *query, const DataType_ *key, const DataType_ *value,
    const DataType_ *key_padding_mask, DataType_ *attn_output, void *buf,
    const int batch_size, const int from_seq_len, const int to_seq_len,
    cublasHandle_t cublas_handle, cudaStream_t stream) {
  DataType_ *Q_buf = (DataType_ *)buf;
  DataType_ *K_buf = (DataType_ *)Q_buf + q_buf_size_;
  DataType_ *V_buf = (DataType_ *)K_buf + k_buf_size_;
  DataType_ *dst_buf = (DataType_ *)V_buf + k_buf_size_;

  dense_layer_kernel_launcher(
      query, param_.query_weight, Q_buf, batch_size * from_seq_len, hidden_dim_,
      hidden_dim_, cublas_handle, stream, param_.cublas_Algo[0]);

  // dense_layer_kernel_launcher(
  //     query, param_.key_weight, K_buf,
  //     batch_size * to_seq_len, hidden_dim_, hidden_dim_, cublas_handle,
  //     stream, param_.cublas_Algo[0]);

  // dense_layer_kernel_launcher(
  //     query, param_.value_weight, V_buf,
  //     batch_size * to_seq_len, hidden_dim_, hidden_dim_, cublas_handle,
  //     stream, param_.cublas_Algo[0]);

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  int M = batch_size * to_seq_len, K = hidden_dim_, N = hidden_dim_;
  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
      param_.key_weight, Traits_::BType, N,
      param_.value_weight - param_.key_weight, key, Traits_::AType, K, 0, &beta,
      K_buf, Traits_::CType, N, k_buf_size_, 2, Traits_::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[1])));

  // int size_per_head = hidden_dim_ / head_num_;
  DataType_ scaler = (DataType_)0.25f; //(1.0f / sqrt(size_per_head));

  dim3 grid(batch_size * head_num_);
  dim3 block;
  if (OpType_ == OperationType::FP32)
    block.x = 128;
  else
    block.x = 128;

  attention_kernel<16><<<grid, block, 0, stream>>>(
      Q_buf, param_.query_bias, K_buf, param_.key_bias, V_buf,
      param_.value_bias, key_padding_mask, dst_buf, batch_size, head_num_,
      from_seq_len, to_seq_len, scaler);

  dense_layer_kernel_launcher(
      dst_buf, param_.out_proj_weight, attn_output, batch_size * from_seq_len,
      hidden_dim_, hidden_dim_, cublas_handle, stream, param_.cublas_Algo[0]);

  grid.x = batch_size * from_seq_len;
  block.x = hidden_dim_; // assert block.x <= 1024

  add_bias_act<ActType::No, DataType_>
      <<<grid, block, 0, stream>>>(attn_output, param_.out_proj_bias,
                                   batch_size * from_seq_len, hidden_dim_);
}

template void MultiHeadAttention<OperationType::FP32>::fused_infer(
    const float *query, const float *key, const float *value,
    const float *key_padding_mask, float *attn_output, void *buf,
    const int batch_size, const int from_seq_len, const int to_seq_len,
    cublasHandle_t cublas_handle, cudaStream_t stream);

template void MultiHeadAttention<OperationType::HALF>::fused_infer(
    const __half *query, const __half *key, const __half *value,
    const __half *key_padding_mask, __half *attn_output, void *buf,
    const int batch_size, const int from_seq_len, const int to_seq_len,
    cublasHandle_t cublas_handle, cudaStream_t stream);
} // namespace fastertransformerv3
