/*
 * Author: Xiaoying Jia, Changyi Wan, Song Yu
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi, yusong.andy}@bytedance.com
 */
#include "fastertransformer_v3/includes/attention.h"
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/disentangle.h"
#include "fastertransformer_v3/includes/nofused_utils.h"
#include "fastertransformer_v3/includes/operators.cuh"
#include "fastertransformer_v3/includes/softmax_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
using namespace std;

namespace fastertransformerv3 {

template <OperationType OpType>
void Attention<OpType>::nofused_infer(
    const DataType_ *query_in, const DataType_ *key_in,
    const DataType_ *value_in, const DataType_ *atten_mask,
    DataType_ *attention_output, void *buf, const int batch_size,
    const int seq_len, cublasHandle_t cublas_handle, cudaStream_t stream) {
  int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;
  int qk_buf_size = ((batch_size * head_num_ * seq_len * seq_len + 15) >> 4)
                    << 4;

  DataType_ *query = (DataType_ *)buf + 0 * input_tensor_size;
  DataType_ *key = (DataType_ *)buf + 1 * input_tensor_size;
  DataType_ *value = (DataType_ *)buf + 2 * input_tensor_size;
  DataType_ *qk_buf = (DataType_ *)buf + 3 * input_tensor_size;
  DataType_ *transpose_dst = (DataType_ *)qk_buf + qk_buf_size;

  DataType_ *attn_score =
      is_deberta_ ? (transpose_dst + input_tensor_size) : nullptr;
  DataType_ *disentangled_buf =
      is_deberta_ ? (attn_score + qk_buf_size) : nullptr;

  int size_per_head_half = (OpType == OperationType::HALF)
                               ? size_per_head_ / 2
                               : size_per_head_; // Be careful.

  dim3 grid, block;

  grid.x = batch_size * seq_len;
  block.x = head_num_ * size_per_head_half;
  add_QKV_bias<<<grid, block, 0, stream>>>(
      query_in, param_.attr_bias_Q, key_in, param_.attr_bias_K, value_in,
      param_.attr_bias_V, query, key, value, batch_size, seq_len, head_num_,
      size_per_head_half);

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

  if (is_deberta_)
    alpha = (DataType_)(1.0f / sqrtf(size_per_head_ *
                                     param_.disentangle_param.scale * 1.0f));

  int M = seq_len, K = size_per_head_, N = seq_len;
  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, key,
      Traits_::BType, K, K * N, query, Traits_::AType, K, M * K, &beta, qk_buf,
      Traits_::CType, N, M * N, batch_size * head_num_, Traits_::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[0])));

  DataType_ *attr_probs;
  if (is_deberta_) {
    disentangle_layer_->infer(qk_buf, query, key, param_.attr_bias_Q,
                              param_.attr_bias_K, attn_score, disentangled_buf,
                              batch_size, seq_len, cublas_handle, stream);

    bool no_scale = true;

    softmax_kernelLauncher<OpType, DataType_>(attn_score, atten_mask,
                                              batch_size, seq_len, head_num_,
                                              size_per_head_, stream, no_scale);

    attr_probs = attn_score;
  } else {
    softmax_kernelLauncher<OpType, DataType_>(qk_buf, atten_mask, batch_size,
                                              seq_len, head_num_,
                                              size_per_head_, stream);
    attr_probs = qk_buf;
  }

  alpha = (DataType_)(1.0f);

  M = seq_len, K = seq_len, N = size_per_head_;
  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, value,
      Traits_::BType, N, K * N, attr_probs, Traits_::AType, K, M * K, &beta,
      transpose_dst, Traits_::CType, N, M * N, batch_size * head_num_,
      Traits_::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[1])));

  grid.x = batch_size * seq_len;
  block.x = size_per_head_half, block.y = head_num_;
  transpose<<<grid, block, 0, stream>>>(transpose_dst, attention_output,
                                        batch_size, seq_len, head_num_,
                                        size_per_head_half);
}

template <OperationType OpType>
void Attention<OpType>::et_nofused_infer(
    const DataType_ *query_in, const DataType_ *key_in,
    const DataType_ *value_in, const DataType_ *atten_mask,
    DataType_ *attention_output, void *buf, const int batch_size,
    const int seq_len, cublasHandle_t cublas_handle, cudaStream_t stream,
    ET_Param et_param) {
  int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;
  int qk_buf_size = ((batch_size * head_num_ * seq_len * seq_len + 15) >> 4)
                    << 4;

  DataType_ *query = (DataType_ *)buf + 0 * input_tensor_size;
  DataType_ *key = (DataType_ *)buf + 1 * input_tensor_size;
  DataType_ *value = (DataType_ *)buf + 2 * input_tensor_size;
  DataType_ *qk_buf = (DataType_ *)buf + 3 * input_tensor_size;
  DataType_ *transpose_dst = qk_buf + qk_buf_size;
  DataType_ *attn_score = transpose_dst + input_tensor_size;
  DataType_ *disentangled_buf = attn_score + qk_buf_size;

  int size_per_head_half = (OpType == OperationType::HALF)
                               ? size_per_head_ / 2
                               : size_per_head_; // Be careful.

  cudaMemsetAsync(query, 0, 3 * input_tensor_size * sizeof(DataType_),
                  stream); // clean zero for batch_gemm

  dim3 grid, block;

  grid.x = et_param.valid_word_num;
  block.x = head_num_ * size_per_head_half;
  add_QKV_bias_padding<<<grid, block, 0, stream>>>(
      query_in, param_.attr_bias_Q, key_in, param_.attr_bias_K, value_in,
      param_.attr_bias_V, query, key, value, batch_size, seq_len, head_num_,
      size_per_head_half, et_param.batch_idx, et_param.word_idx);

  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
  int M = seq_len, K = size_per_head_, N = seq_len;

  if (is_deberta_)
    alpha = (DataType_)(1.0f / sqrtf(size_per_head_ *
                                     param_.disentangle_param.scale * 1.0f));

  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, key,
      Traits_::BType, K, K * N, query, Traits_::AType, K, M * K, &beta, qk_buf,
      Traits_::CType, N, M * N, batch_size * head_num_, Traits_::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[0])));

  DataType_ *attr_probs;
  if (is_deberta_) {
    disentangle_layer_->infer(qk_buf, query, key, param_.attr_bias_Q,
                              param_.attr_bias_K, attn_score, disentangled_buf,
                              batch_size, seq_len, cublas_handle, stream);

    bool no_scale = true;
    softmax_kernelLauncher<OpType, DataType_>(attn_score, atten_mask,
                                              batch_size, seq_len, head_num_,
                                              size_per_head_, stream, no_scale);

    attr_probs = attn_score;

  } else {
    softmax_kernelLauncher<OpType, DataType_>(qk_buf, atten_mask, batch_size,
                                              seq_len, head_num_,
                                              size_per_head_, stream);
    attr_probs = qk_buf;
  }

  alpha = (DataType_)(1.0f);
  M = seq_len, K = seq_len, N = size_per_head_;
  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, value,
      Traits_::BType, N, K * N, attr_probs, Traits_::AType, K, M * K, &beta,
      transpose_dst, Traits_::CType, N, M * N, batch_size * head_num_,
      Traits_::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[1])));

  grid.x = et_param.valid_word_num;
  block.x = size_per_head_half, block.y = head_num_;
  transpose_rm_padding<<<grid, block, 0, stream>>>(
      transpose_dst, attention_output, batch_size, seq_len, head_num_,
      size_per_head_half, et_param.batch_idx, et_param.word_idx);
}

template void Attention<OperationType::FP32>::nofused_infer(
    const float *query, const float *key, const float *value,
    const float *atten_mask, float *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream);

template void Attention<OperationType::HALF>::nofused_infer(
    const __half *query, const __half *key, const __half *value,
    const __half *atten_mask, __half *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream);

template void Attention<OperationType::FP32>::et_nofused_infer(
    const float *query, const float *key, const float *value,
    const float *atten_mask, float *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream, ET_Param et_param);

template void Attention<OperationType::HALF>::et_nofused_infer(
    const __half *query, const __half *key, const __half *value,
    const __half *atten_mask, __half *attention_output, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream, ET_Param et_param);
} // namespace fastertransformerv3
