/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/conformer.h"

namespace fastertransformerv3 {

template <OperationType OpType_> class OCR_ConformerParam {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  DataType_ *pos_encoder_src;
  ConformerParam<OpType_> *conformer_param;

  // int cublas_Algo[3];

  OCR_ConformerParam() {
    pos_encoder_src = nullptr;

    // if(OpType_ == OperationType::HALF)
    //     cublas_Algo[0] = 99, cublas_Algo[1] = 99, cublas_Algo[2] = 99;
    // else
    //     cublas_Algo[0] = -1, cublas_Algo[1] = -1, cublas_Algo[2] = -1;
  }
};

template <typename T> struct OCR_ConformerInferParam {
  const T *input_tensor;
  const T *atten_mask;
  T *transformer_output;
  void *buf;
  int batch_size;
  int seq_len;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_> class OCR_Conformer {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

  OCR_ConformerParam<OpType_> param_;
  using OCR_ConformerInferParam = struct OCR_ConformerInferParam<DataType_>;

  Conformer<OpType_> **conformer_layer_ = nullptr;

  const int max_batch_size_, max_seq_len_, head_num_, size_per_head_;

  unsigned long long inner_buf_size_;

  bool use_fused_attention_, is_remove_padding_;
  const bool use_fp32_; // gamma & beta datatype

  const int layers_;

public:
  OCR_Conformer(const int max_batch_size, const int head_num,
                const int size_per_head, const int max_seq_len,
                const int layers, const bool use_fused_attention = true,
                const bool is_remove_padding = true,
                const bool use_fp32 = false)
      : max_batch_size_(max_batch_size), head_num_(head_num),
        size_per_head_(size_per_head), max_seq_len_(max_seq_len),
        layers_(layers), use_fused_attention_(use_fused_attention),
        is_remove_padding_(is_remove_padding), use_fp32_(use_fp32) {
    conformer_layer_ =
        (Conformer<OpType_> **)malloc(layers_ * sizeof(Conformer<OpType_> *));
    for (int i = 0; i < layers_; i++)
      conformer_layer_[i] = new Conformer<OpType_>(
          max_batch_size, head_num, size_per_head, max_seq_len,
          use_fused_attention_, is_remove_padding_, use_fp32_);
  }

  unsigned long long cal_bufsize() {
    // inner_buf_size_ = max_batch_size_ * max_seq_len_ * head_num_ *
    // size_per_head_;
    unsigned long long total_buf_size = conformer_layer_[0]->cal_bufsize();
    return total_buf_size;
  }

  void initialize(OCR_ConformerParam<OpType_> param) {
    param_ = param;
    for (int i = 0; i < layers_; i++)
      conformer_layer_[i]->initialize(param_.conformer_param[i]);
  }

  void infer(OCR_ConformerInferParam infer_param);

  ~OCR_Conformer() {
    for (int i = 0; i < layers_; i++)
      delete conformer_layer_[i];
    free(conformer_layer_);
  }
};
} // namespace fastertransformerv3