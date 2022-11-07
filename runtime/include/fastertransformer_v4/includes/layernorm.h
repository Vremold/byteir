/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <OperationType OpType> class LayerNormParam {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  const void *gamma;
  const void *beta;

  LayerNormParam() {
    gamma = nullptr;
    beta = nullptr;
  }
};

template <typename T> struct LayerNormForwardParam {
  const T *input;
  T *mean;
  T *var_rsqrt;
  T *layernorm_out;
  const int rows;
  cudaStream_t stream;
  const T *residual = NULL;
  T *input_add_residual = NULL;
};

template <typename T> struct LayerNormBackwardParam {
  const T *grad_out;
  const T *input_add_residual; // = input if residual is NULL
  const T *mean;
  const T *var_rsqrt;
  T *grad_in;
  T *grad_gamma;
  T *grad_beta;
  void *buf;
  const int rows;
  cudaStream_t stream;
  T *grad_residual = NULL;
};

template <OperationType OpType> class LayerNorm {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  LayerNormParam<OpType> param_;
  using LayerNormForwardParam = struct LayerNormForwardParam<DataType_>;
  using LayerNormBackwardParam = struct LayerNormBackwardParam<DataType_>;

  const int max_rows_, hidden_dim_;
  unsigned long long fw_inner_buf_size_ = 0, bw_inner_buf_size_ = 0;
  const bool use_fp32_; // gamma & beta datatype

  const int block_count_ = 320; // 320 for V100, 160 for T4
public:
  LayerNorm(const int max_rows, const int hidden_dim,
            const bool use_fp32 = false)
      : max_rows_(max_rows), hidden_dim_(hidden_dim), use_fp32_(use_fp32) {}

  unsigned long long cal_fw_bufsize() {
    fw_inner_buf_size_ = 0;
    return fw_inner_buf_size_;
  }

  unsigned long long cal_bw_bufsize() {
    bw_inner_buf_size_ = block_count_ * hidden_dim_ * sizeof(float) * 2;
    return bw_inner_buf_size_;
  }

  void initialize(LayerNormParam<OpType> param) { param_ = param; }

  void forward(LayerNormForwardParam fw_param);

  void backward(LayerNormBackwardParam bw_param);

  ~LayerNorm() {}
};
} // namespace fastertransformerv4