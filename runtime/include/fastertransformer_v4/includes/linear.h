/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <OperationType OpType> class LinearParam {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  const DataType_ *weight;
  const DataType_ *bias;

  LinearParam() {
    weight = nullptr;
    bias = nullptr;
  }
};

template <typename T> struct LinearForwardParam {
  const T *input;
  T *output;
  const int rows;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  T *bias_out = NULL;
  uint8_t *dropout_mask = NULL;
};

template <typename T> struct LinearBackwardParam {
  const T *grad_out;
  const T *input;
  T *grad_in;
  T *grad_weight;
  T *grad_bias;
  void *buf;
  const int rows;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  T *bias_out = NULL;
  uint8_t *dropout_mask = NULL;
};

template <OperationType OpType> class Linear {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  LinearParam<OpType> param_;
  using LinearForwardParam = struct LinearForwardParam<DataType_>;
  using LinearBackwardParam = struct LinearBackwardParam<DataType_>;

  const int max_rows_, K_, N_;
  const bool transposed_weight_;
  const bool act_gelu_;
  const float dropout_rate_;
  unsigned long long fw_inner_buf_size_ = 0, bw_inner_buf_size_ = 0;

  const int block_count_ = 320; // 320 for V100, 160 for T4
public:
  Linear(const int max_rows, const int K, const int N,
         const bool transposed_weight = false, const bool act_gelu = false,
         const float dropout_rate = 0.0f)
      : max_rows_(max_rows), K_(K), N_(N),
        transposed_weight_(transposed_weight), act_gelu_(act_gelu),
        dropout_rate_(dropout_rate) {}

  unsigned long long cal_fw_bufsize() {
    fw_inner_buf_size_ = 0;
    return fw_inner_buf_size_;
  }

  unsigned long long cal_bw_bufsize() {
    bw_inner_buf_size_ = block_count_ * N_ * sizeof(float);
    return bw_inner_buf_size_;
  }

  void initialize(LinearParam<OpType> param) { param_ = param; }

  void forward(LinearForwardParam fw_param);
  void backward(LinearBackwardParam bw_param);

  ~Linear() {}
};
} // namespace fastertransformerv4