/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/common.h"

namespace fastertransformerv3 {

template <OperationType OpType_> class MultiHeadAttentionParam {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  const DataType_ *query_weight;
  const DataType_ *query_bias;
  const DataType_ *key_weight;
  const DataType_ *key_bias;
  const DataType_ *value_weight;
  const DataType_ *value_bias;

  const DataType_ *out_proj_weight;
  const DataType_ *out_proj_bias;

  int cublas_Algo[2];

  MultiHeadAttentionParam() {
    query_weight = nullptr;
    query_bias = nullptr;
    key_weight = nullptr;
    key_bias = nullptr;
    value_weight = nullptr;
    value_bias = nullptr;

    out_proj_weight = nullptr;
    out_proj_bias = nullptr;

    if (OpType_ == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1;
  }
};

template <OperationType OpType_> class MultiHeadAttention {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const int max_batch_size_, max_from_seq_len_, max_to_seq_len_, hidden_dim_,
      head_num_;
  MultiHeadAttentionParam<OpType_> param_;

  int q_buf_size_, k_buf_size_, dst_buf_size_;

  const bool use_fused_attention_;

public:
  MultiHeadAttention(const int max_batch_size, const int max_from_seq_len,
                     const int max_to_seq_len, const int hidden_dim,
                     const int head_num, const bool use_fused_attention = true)
      : max_batch_size_(max_batch_size), max_from_seq_len_(max_from_seq_len),
        max_to_seq_len_(max_to_seq_len), hidden_dim_(hidden_dim),
        head_num_(head_num), use_fused_attention_(use_fused_attention) {}
  unsigned long long cal_bufsize() {
    q_buf_size_ = (max_batch_size_ * max_from_seq_len_) * hidden_dim_;
    k_buf_size_ = (max_batch_size_ * max_to_seq_len_) * hidden_dim_;
    dst_buf_size_ = (max_batch_size_ * max_from_seq_len_) * hidden_dim_;

    unsigned long long inner_buf_size_ =
        q_buf_size_ + k_buf_size_ * 2 + dst_buf_size_;

    // if(use_fused_attention_ == false)
    // {
    //     unsigned long long input_tensor_size = max_batch_size_ * head_num_ *
    //     max_seq_len_ * size_per_head_; unsigned long long qk_buf_size =
    //     ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >>
    //     4) << 4; inner_buf_size_ += input_tensor_size + qk_buf_size;
    // }

    // inner_buf_size_ = ((inner_buf_size_ * sizeof(DataType_) + 31) >> 5) << 5;
    return inner_buf_size_ * sizeof(DataType_);
  }
  void initialize(MultiHeadAttentionParam<OpType_> param) { param_ = param; }

  void infer(const DataType_ *query, const DataType_ *key,
             const DataType_ *value, const DataType_ *key_padding_mask,
             DataType_ *attention_output, void *buf, const int batch_size,
             const int from_seq_len, const int to_seq_len,
             cublasHandle_t cublas_handle, cudaStream_t stream) {

    // if(use_fused_attention_)
    fused_infer(query, key, value, key_padding_mask, attention_output, buf,
                batch_size, from_seq_len, to_seq_len, cublas_handle, stream);
    // else
    //     nofused_infer(
    //         query, key, value, key_padding_mask, attention_output, buf,
    //         batch_size, from_seq_len, to_seq_len, cublas_handle, stream);
  }

  // void nofused_infer(
  //     const DataType_* query, const DataType_* key, const DataType_* value,
  //     const DataType_* key_padding_mask, DataType_* attention_output, void*
  //     buf, const int batch_size, const int from_seq_len, const int
  //     to_seq_len, cublasHandle_t cublas_handle, cudaStream_t stream);

  void fused_infer(const DataType_ *query, const DataType_ *key,
                   const DataType_ *value, const DataType_ *key_padding_mask,
                   DataType_ *attn_output, void *buf, const int batch_size,
                   const int from_seq_len, const int to_seq_len,
                   cublasHandle_t cublas_handle, cudaStream_t stream);

  ~MultiHeadAttention() {}
};
} // namespace fastertransformerv3