/*
 * Author: Xiaoying Jia, Song Yu
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, yusong.andy}@bytedance.com
 */
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/disentangle.h"
#include "fastertransformer_v3/includes/utils.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformerv3 {

template <typename T>
__global__ void relative_embedding_lookup(T *relative_embedding_out,
                                          const T *relative_embedding,
                                          const int start, const int end,
                                          const int dim) {
  int block_idx = blockIdx.x;
  if (block_idx + start < end) {
    const T *block_from = relative_embedding + dim * (block_idx + start);
    T *block_to = relative_embedding_out + dim * block_idx;
    int tid = threadIdx.x;
    for (int i = tid; i < dim; i += blockDim.x) {
      block_to[i] = block_from[i];
    }
  }
}

template <typename T>
void relative_embedding_lookup_kernel_launcher(T *relative_embedding_res,
                                               const T *relative_embedding,
                                               const int seq_len,
                                               const int max_relative_positions,
                                               const int hidden_dim,
                                               cudaStream_t stream) {
  int start = max_relative_positions - seq_len,
      end = max_relative_positions + seq_len;
  dim3 grid(seq_len * 2);
  dim3 block(1024);
  relative_embedding_lookup<<<grid, block, 0, stream>>>(
      relative_embedding_res, relative_embedding, start, end, hidden_dim);
}

template <typename T>
__global__ void
add_bias_transpose(T *trans_pos_query_buf, T *trans_pos_key_buf,
                   const T *pos_query_buf, const T *pos_key_buf,
                   const T *bias_query, const T *bias_key,
                   // const T *pos_query_buf, const T *pos_key_buf, const T
                   // *bias_key, const T *bias_query,
                   const int batch_size, const int seq_len, const int head_num,
                   const int size_per_head) {
  int seq_id = blockIdx.x / head_num;
  int head_id = blockIdx.x % head_num;
  int dim_id = threadIdx.x;

  int src_id = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_id = head_id * size_per_head + dim_id;
  int tgt_id =
      head_id * (seq_len * size_per_head) + seq_id * size_per_head + dim_id;

  T pos_query_value =
      __ldg(pos_query_buf + src_id) + __ldg(bias_query + bias_id);
  T pos_key_value = __ldg(pos_key_buf + src_id) + __ldg(bias_key + bias_id);

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    int batch_tgt_id = tgt_id + batch_id * (head_num * seq_len * size_per_head);
    trans_pos_query_buf[batch_tgt_id] = pos_query_value;
    trans_pos_key_buf[batch_tgt_id] = pos_key_value;
  }
}

template <typename T>
void add_bias_transpose_kernel_launcher(
    T *trans_pos_query_buf, const T *pos_query_buf, const T *bias_query,
    T *trans_pos_key_buf, const T *pos_key_buf, const T *bias_key,
    int batch_size, int seq_len, int head_num, int size_per_head,
    cudaStream_t stream) {
  dim3 grid(1 * seq_len * head_num), block(size_per_head);
  add_bias_transpose<<<grid, block, 0, stream>>>(
      trans_pos_query_buf, trans_pos_key_buf, pos_query_buf, pos_key_buf,
      bias_query, bias_key, batch_size, seq_len, head_num, size_per_head);
}

template <typename T>
__global__ void gather_torch_kernel(const T *p2c, const T *c2p, const T *score,
                                    T *final, const int batch_size,
                                    const int num_heads, const int seq_len,
                                    const T scaler, const bool is_paper) {

  int bid = blockIdx.x;
  int offset = bid * seq_len * 2 * seq_len;

  for (int tid = threadIdx.x; tid < seq_len * seq_len; tid += blockDim.x) {
    int i = tid / seq_len;
    int j = tid % seq_len;

    int c2p_index = offset + i * seq_len * 2 + seq_len - 1 + i - j;
    int p2c_index =
        is_paper ? offset + j * seq_len * 2 + seq_len - 1 - i + j : c2p_index;

    T p2c_val = __ldg(&p2c[p2c_index]);
    T c2p_val = __ldg(&c2p[c2p_index]);

    T sum = (p2c_val + c2p_val) * scaler;
    sum += __ldg(&score[bid * seq_len * seq_len + tid]);
    final[bid * seq_len * seq_len + tid] = sum;
  }
}

template <OperationType OpType_>
void Disentangle<OpType_>::infer(
    const DataType_ *attn_score, const DataType_ *query_out,
    const DataType_ *key_out, const DataType_ *query_bias,
    const DataType_ *key_bias, DataType_ *attn_score_out, void *buf,
    const int batch_size, const int seq_len, cublasHandle_t cublas_handle,
    cudaStream_t stream) {
  const int input_tensor_size = seq_len * 2 * head_num_ * size_per_head_;
  const int middle_tensor_size =
      ((batch_size * head_num_ * seq_len * seq_len * 2 + 15) >> 4) << 4;

  DataType_ *relative_embedding_res = (DataType_ *)(buf);
  DataType_ *pos_query_buf = relative_embedding_res + input_tensor_size;
  DataType_ *pos_key_buf = pos_query_buf + input_tensor_size;
  DataType_ *trans_pos_query_buf = pos_key_buf + input_tensor_size;
  DataType_ *trans_pos_key_buf =
      trans_pos_query_buf + batch_size * input_tensor_size;
  DataType_ *c2p_att = trans_pos_key_buf + batch_size * input_tensor_size;
  DataType_ *p2c_att = c2p_att + middle_tensor_size;

  // [1, seq_len*2, head_num * size_per_head]
  relative_embedding_lookup_kernel_launcher(
      relative_embedding_res, param_.relative_embedding, seq_len,
      param_.max_pos, head_num_ * size_per_head_, stream);

  // [1, seq_len*2, head_num * size_per_head]

  int m = 1 * seq_len * 2;
  int k = head_num_ * size_per_head_;
  int n = k;
  dense_layer_kernel_launcher(relative_embedding_res, param_.attr_kernel_Q,
                              pos_query_buf, m, k, n, cublas_handle, stream,
                              param_.cublas_Algo[0]);

  dense_layer_kernel_launcher(relative_embedding_res, param_.attr_kernel_K,
                              pos_key_buf, m, k, n, cublas_handle, stream,
                              param_.cublas_Algo[0]);

  // print_vec(query_bias, "query_bias", 10);
  // print_vec(key_bias, "key_bias", 10);
  add_bias_transpose_kernel_launcher(trans_pos_query_buf, pos_query_buf,
                                     query_bias, trans_pos_key_buf, pos_key_buf,
                                     key_bias, batch_size, seq_len * 2,
                                     head_num_, size_per_head_, stream);
  // [batch_size, head_num, seq_len, seq_len*2]
  int N = seq_len * 2;
  int M = seq_len;
  int K = size_per_head_;
  DataType_ scaler =
      (DataType_)(1.0f / sqrtf(size_per_head_ * param_.scale * 1.0f));
  DataType_ pos_query_alpha = param_.is_paper ? (DataType_)(1.0f) : scaler;
  DataType_ pos_key_alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;

  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &pos_key_alpha,
      trans_pos_key_buf, Traits<OpType_>::BType, K, K * N, query_out,
      Traits<OpType_>::AType, K, M * K, &beta, c2p_att, Traits<OpType_>::CType,
      N, M * N, batch_size * head_num_, Traits<OpType_>::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[1])));

  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &pos_query_alpha,
      trans_pos_query_buf, Traits<OpType_>::BType, K, K * N, key_out,
      Traits<OpType_>::AType, K, M * K, &beta, p2c_att, Traits<OpType_>::CType,
      N, M * N, batch_size * head_num_, Traits<OpType_>::computeType,
      static_cast<cublasGemmAlgo_t>(param_.cublas_Algo[1])));

  dim3 grid(batch_size * head_num_);
  dim3 block(1024);

  if (!param_.is_paper)
    scaler = (DataType_)(1.0f);

  gather_torch_kernel<<<grid, block, 0, stream>>>(
      p2c_att, c2p_att, attn_score, attn_score_out, batch_size, head_num_,
      seq_len, scaler, param_.is_paper);
}

template void Disentangle<OperationType::FP32>::infer(
    const float *attn_score, const float *query_out, const float *key_out,
    const float *query_bias, const float *key_bias, float *attn_score_out,
    void *buf, const int batch_size, const int seq_len,
    cublasHandle_t cublas_handle, cudaStream_t stream);

template void Disentangle<OperationType::HALF>::infer(
    const __half *attn_score, const __half *query_out, const __half *key_out,
    const __half *query_bias, const __half *key_bias, __half *attn_score_out,
    void *buf, const int batch_size, const int seq_len,
    cublasHandle_t cublas_handle, cudaStream_t stream);
} // namespace fastertransformerv3
