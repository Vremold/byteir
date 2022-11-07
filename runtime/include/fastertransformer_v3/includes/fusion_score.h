/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/fused_multi_head_attention.h"

namespace fastertransformerv3 {

template <OperationType OpType_> class FusionScoreParam {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  MultiHeadAttentionParam<OpType_> multi_head_attention_param;

  const DataType_ *linear1_weight;
  const DataType_ *linear1_bias;
  const DataType_ *linear2_weight;
  const DataType_ *linear2_bias;

  int cublas_Algo[2];

  FusionScoreParam() {
    linear1_weight = nullptr;
    linear1_bias = nullptr;
    linear2_weight = nullptr;
    linear2_bias = nullptr;

    if (OpType_ == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1;
  }
};

template <OperationType OpType_> class FusionScore {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const int max_batch_size_, max_from_seq_len_, max_to_seq_len_, hidden_dim_,
      head_num_;
  const int fc_hidden_size1_, fc_hidden_size2_;
  FusionScoreParam<OpType_> param_;

  MultiHeadAttention<OpType_> *multi_head_attention_layer_ = nullptr;

  int key_T_size_, attention_output_size_, mid_out_size_;

  const bool use_fused_attention_;

public:
  FusionScore(const int max_batch_size, const int max_from_seq_len,
              const int max_to_seq_len, const int hidden_dim,
              const int head_num, const int fc_hidden_size1,
              const int fc_hidden_size2, const bool use_fused_attention = true)
      : max_batch_size_(max_batch_size), max_from_seq_len_(max_from_seq_len),
        max_to_seq_len_(max_to_seq_len), hidden_dim_(hidden_dim),
        head_num_(head_num), fc_hidden_size1_(fc_hidden_size1),
        fc_hidden_size2_(fc_hidden_size2),
        use_fused_attention_(use_fused_attention) {
    multi_head_attention_layer_ = new MultiHeadAttention<OpType_>(
        max_batch_size_, max_from_seq_len_, max_to_seq_len_, hidden_dim_,
        head_num_, use_fused_attention_);
  }
  unsigned long long cal_bufsize() {
    key_T_size_ = (max_to_seq_len_ * max_batch_size_) * hidden_dim_;
    attention_output_size_ =
        (max_batch_size_ * max_from_seq_len_) * hidden_dim_;
    mid_out_size_ = (max_batch_size_ * max_from_seq_len_) * fc_hidden_size1_;

    unsigned long long inner_buf_size_ =
        key_T_size_ + attention_output_size_ + mid_out_size_;

    // if(use_fused_attention_ == false)
    // {
    //     unsigned long long input_tensor_size = max_batch_size_ * head_num_ *
    //     max_seq_len_ * size_per_head_; unsigned long long qk_buf_size =
    //     ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >>
    //     4) << 4; inner_buf_size_ += input_tensor_size + qk_buf_size;
    // }

    inner_buf_size_ = ((inner_buf_size_ * sizeof(DataType_) + 31) >> 5)
                      << 5; // For 32B memory alignment

    unsigned long long total_buf_size =
        multi_head_attention_layer_->cal_bufsize() + inner_buf_size_;

    return total_buf_size;
  }
  void initialize(FusionScoreParam<OpType_> param) {
    param_ = param;
    multi_head_attention_layer_->initialize(param_.multi_head_attention_param);
  }

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

  ~FusionScore() { delete multi_head_attention_layer_; }
};
} // namespace fastertransformerv3