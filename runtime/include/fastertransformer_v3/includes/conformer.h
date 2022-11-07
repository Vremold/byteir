/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Inference
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once

#include "fastertransformer_v3/includes/attention.h"
#include "fastertransformer_v3/includes/common.h"
#include "fastertransformer_v3/includes/convolution.h"

namespace fastertransformerv3 {

template <OperationType OpType_> class ConformerParam {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  const void *ffn1_layernorm_gamma;
  const void *ffn1_layernorm_beta;
  const DataType_ *ffn1_inter_kernel;
  const DataType_ *ffn1_inter_bias;
  const DataType_ *ffn1_output_kernel;
  const DataType_ *ffn1_output_bias;

  const void *attr_output_layernorm_gamma;
  const void *attr_output_layernorm_beta;
  const DataType_ *attr_kernel_Q;
  const DataType_ *attr_kernel_K;
  const DataType_ *attr_kernel_V;
  AttentionParam<OpType_> attention_param;
  const DataType_ *attr_output_kernel;
  const DataType_ *attr_output_bias;

  const void *conv_layernorm_gamma;
  const void *conv_layernorm_beta;
  const DataType_ *pointwise_conv_kernel_1;
  const DataType_ *pointwise_conv_bias_1;
  const DataType_ *depthwise_conv_kernel;
  const DataType_ *depthwise_conv_bias;
  const DataType_ *batchnorm_mean;
  const DataType_ *batchnorm_var;
  const void *batchnorm_gamma;
  const void *batchnorm_beta;
  const DataType_ *pointwise_conv_kernel_2;
  const DataType_ *pointwise_conv_bias_2;

  const void *output_layernorm_gamma;
  const void *output_layernorm_beta;
  const DataType_ *inter_kernel;
  const DataType_ *inter_bias;
  const DataType_ *output_kernel;
  const DataType_ *output_bias;

  const void *last_layernorm_gamma;
  const void *last_layernorm_beta;

  int cublas_Algo[3];

  ConformerParam() {
    ffn1_layernorm_gamma = nullptr;
    ffn1_layernorm_beta = nullptr;
    ffn1_inter_kernel = nullptr;
    ffn1_inter_bias = nullptr;
    ffn1_output_kernel = nullptr;
    ffn1_output_bias = nullptr;

    attr_output_layernorm_gamma = nullptr;
    attr_output_layernorm_beta = nullptr;
    attr_kernel_Q = nullptr;
    attr_kernel_K = nullptr;
    attr_kernel_V = nullptr;
    attr_output_kernel = nullptr;
    attr_output_bias = nullptr;

    output_layernorm_gamma = nullptr;
    output_layernorm_beta = nullptr;
    inter_kernel = nullptr;
    inter_bias = nullptr;
    output_kernel = nullptr;
    output_bias = nullptr;

    if (OpType_ == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99, cublas_Algo[2] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1, cublas_Algo[2] = -1;
  }
};

template <typename T> struct ConformerInferParam {
  const T *input_tensor;
  const T *atten_mask;
  T *transformer_output;
  void *buf;
  int batch_size;
  int seq_len;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_> class Conformer {
private:
  typedef Traits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;

  ConformerParam<OpType_> param_;
  using ConformerInferParam = struct ConformerInferParam<DataType_>;

  Attention<OpType_> *attention_layer_ = nullptr;

  const int max_batch_size_, head_num_, size_per_head_, max_seq_len_;

  unsigned long long inner_buf_size_;

  bool use_fused_attention_, is_remove_padding_;
  const bool use_fp32_; // gamma & beta datatype

public:
  Conformer(const int max_batch_size, const int head_num,
            const int size_per_head, const int max_seq_len,
            const bool use_fused_attention = true,
            const bool is_remove_padding = true, const bool use_fp32 = false)
      : max_batch_size_(max_batch_size), head_num_(head_num),
        size_per_head_(size_per_head), max_seq_len_(max_seq_len),
        use_fused_attention_(use_fused_attention),
        is_remove_padding_(is_remove_padding), use_fp32_(use_fp32) {
    if (OpType_ == OperationType::FP32 || size_per_head_ != 64 ||
        max_seq_len_ > 64)
      use_fused_attention_ = false;

    if (max_batch_size_ <= 4)
      is_remove_padding_ = false;

    attention_layer_ = new Attention<OpType_>(
        max_batch_size, head_num, size_per_head, max_seq_len,
        use_fused_attention_, is_remove_padding_);
  }

  unsigned long long cal_bufsize() {
    inner_buf_size_ = 0;

    unsigned long long input_tensor_size =
        max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
    inner_buf_size_ +=
        input_tensor_size * 15; // inter_matmul_buf_ = 4 * input_tensor_size

    inner_buf_size_ *= sizeof(DataType_);

    if (is_remove_padding_) // batch_idx & word_idx
      inner_buf_size_ += max_batch_size_ * max_seq_len_ * 2 * sizeof(int);

    inner_buf_size_ = ((inner_buf_size_ + 31) >> 5)
                      << 5; // For 32B memory alignment

    unsigned long long total_buf_size =
        attention_layer_->cal_bufsize() + inner_buf_size_;

    return total_buf_size;
  }

  void initialize(ConformerParam<OpType_> param) {
    param_ = param;
    attention_layer_->initialize(param_.attention_param);
  }

  void infer(ConformerInferParam infer_param);

  ~Conformer() { delete attention_layer_; }
};
} // namespace fastertransformerv3