/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/disentangle.h"

namespace fastertransformerv3 {

typedef struct {
  int *batch_idx;
  int *word_idx;
  int valid_word_num;
} ET_Param;

template <OperationType OpType> class AttentionParam {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  DisentangleParam<OpType> disentangle_param;

  const DataType_ *attr_bias_Q;
  const DataType_ *attr_bias_K;
  const DataType_ *attr_bias_V;

  int cublas_Algo[2];

  AttentionParam() {
    attr_bias_Q = nullptr;
    attr_bias_K = nullptr;
    attr_bias_V = nullptr;

    if (OpType == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1;
  }
};

template <OperationType OpType> class Attention {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;
  const int max_batch_size_, head_num_, size_per_head_, max_seq_len_;
  AttentionParam<OpType> param_;
  Disentangle<OpType> *disentangle_layer_ = nullptr;

  const bool use_fused_attention_, is_remove_padding_;
  const bool is_deberta_;

public:
  Attention(const int max_batch_size, const int head_num,
            const int size_per_head, const int max_seq_len,
            const bool use_fused_attention = true,
            const bool is_remove_padding = false, const bool is_deberta = false)
      : max_batch_size_(max_batch_size), head_num_(head_num),
        size_per_head_(size_per_head), max_seq_len_(max_seq_len),
        use_fused_attention_(use_fused_attention),
        is_remove_padding_(is_remove_padding), is_deberta_(is_deberta) {
    if (is_deberta_)
      disentangle_layer_ = new Disentangle<OpType>(head_num, size_per_head,
                                                   max_seq_len, max_batch_size);
    // assert(!is_deberta_ || (is_deberta_ && !use_fused_attention_));
  }
  unsigned long long cal_bufsize() {
    if (!is_deberta_) {
      if (use_fused_attention_)
        return 0;
      else {
        unsigned long long input_tensor_size =
            max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
        // for memory alignment
        unsigned long long qk_buf_size =
            ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >>
             4)
            << 4;
        unsigned long long inner_buf_size_ =
            input_tensor_size * 4 + qk_buf_size;
        return inner_buf_size_ * sizeof(DataType_);
      }
    } else {
      unsigned long long input_tensor_size =
          max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
      // for memory alignment
      unsigned long long qk_buf_size =
          ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >>
           4)
          << 4;

      unsigned long long pos_tensor_size =
          max_seq_len_ * 2 * head_num_ * size_per_head_;
      unsigned long long middle_tensor_size =
          ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ * 2 +
            15) >>
           4)
          << 4;
      // memory alignment
      unsigned long long output_tensor_size =
          ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >>
           4)
          << 4;
      unsigned long long inner_buf_size_ =
          input_tensor_size * 4 + qk_buf_size + pos_tensor_size * 3 +
          max_batch_size_ * pos_tensor_size * 2 + middle_tensor_size * 2 +
          output_tensor_size;
      return inner_buf_size_ * sizeof(DataType_);
    }
  }

  void initialize(AttentionParam<OpType> param) {
    param_ = param;
    if (is_deberta_)
      disentangle_layer_->initialize(param.disentangle_param);
  }

  void infer(const DataType_ *query, const DataType_ *key,
             const DataType_ *value, const DataType_ *atten_mask,
             DataType_ *attention_output, void *buf, const int batch_size,
             const int seq_len, cublasHandle_t cublas_handle,
             cudaStream_t stream, ET_Param et_param = {nullptr, nullptr, 0}) {
    if (is_remove_padding_) {
      if (use_fused_attention_)
        et_fused_infer(query, key, value, atten_mask, attention_output, buf,
                       batch_size, seq_len, cublas_handle, stream, et_param);
      else
        et_nofused_infer(query, key, value, atten_mask, attention_output, buf,
                         batch_size, seq_len, cublas_handle, stream, et_param);
    } else {
      if (use_fused_attention_)
        fused_infer(query, key, value, atten_mask, attention_output, buf,
                    batch_size, seq_len, cublas_handle, stream);
      else
        nofused_infer(query, key, value, atten_mask, attention_output, buf,
                      batch_size, seq_len, cublas_handle, stream);
    }
  }

  void nofused_infer(const DataType_ *query, const DataType_ *key,
                     const DataType_ *value, const DataType_ *atten_mask,
                     DataType_ *attention_output, void *buf,
                     const int batch_size, const int seq_len,
                     cublasHandle_t cublas_handle, cudaStream_t stream);

  void fused_infer(const DataType_ *query, const DataType_ *key,
                   const DataType_ *value, const DataType_ *atten_mask,
                   DataType_ *attention_output, void *buf, const int batch_size,
                   const int seq_len, cublasHandle_t cublas_handle,
                   cudaStream_t stream);

  void et_nofused_infer(const DataType_ *query, const DataType_ *key,
                        const DataType_ *value, const DataType_ *atten_mask,
                        DataType_ *attention_output, void *buf,
                        const int batch_size, const int seq_len,
                        cublasHandle_t cublas_handle, cudaStream_t stream,
                        ET_Param et_param);

  void et_fused_infer(const DataType_ *query, const DataType_ *key,
                      const DataType_ *value, const DataType_ *atten_mask,
                      DataType_ *attention_output, void *buf,
                      const int batch_size, const int seq_len,
                      cublasHandle_t cublas_handle, cudaStream_t stream,
                      ET_Param et_param);

  ~Attention() {
    if (is_deberta_)
      delete disentangle_layer_;
  }
};
} // namespace fastertransformerv3
