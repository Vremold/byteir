/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <OperationType OpType> class LinearTransposeParam {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

public:
  const DataType_ *weight;
  const DataType_ *bias;

  LinearTransposeParam() {
    weight = nullptr;
    bias = nullptr;
  }
};

template <typename T> struct LinearTransposeForwardParam {
  const T *input;
  T *output;
  void *buf;
  const int batch_size;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  transposeType transpose_type = TRANSPOSE0213;
};

template <typename T> struct LinearTransposeBackwardParam {
  const T *grad_out;
  const T *input;
  T *grad_in;
  T *grad_weight;
  T *grad_bias;
  void *buf;
  const int batch_size;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  transposeType transpose_type = TRANSPOSE0213;
};

template <OperationType OpType> class LinearTranspose {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  LinearTransposeParam<OpType> param_;
  using LinearTransposeForwardParam =
      struct LinearTransposeForwardParam<DataType_>;
  using LinearTransposeBackwardParam =
      struct LinearTransposeBackwardParam<DataType_>;

  const int max_batch_size_, seq_len_, from_hidden_dim_, to_hidden_dim_,
      head_num_;
  const bool transposed_weight_;
  unsigned long long fw_inner_buf_size_ = 0, bw_inner_buf_size_ = 0;

  const int block_count_ = 320; // 320 for V100, 160 for T4
public:
  LinearTranspose(const int max_batch_size, const int seq_len,
                  const int from_hidden_dim, const int to_hidden_dim,
                  const int head_num, const bool transposed_weight = false)
      : max_batch_size_(max_batch_size), seq_len_(seq_len),
        from_hidden_dim_(from_hidden_dim), to_hidden_dim_(to_hidden_dim),
        head_num_(head_num), transposed_weight_(transposed_weight) {}

  unsigned long long cal_fw_bufsize() {
    fw_inner_buf_size_ = 0;
    fw_inner_buf_size_ +=
        max_batch_size_ * seq_len_ * to_hidden_dim_ * sizeof(DataType_);
    return fw_inner_buf_size_;
  }

  unsigned long long cal_bw_bufsize() {
    bw_inner_buf_size_ = 0;
    bw_inner_buf_size_ += block_count_ * to_hidden_dim_ * sizeof(float);
    bw_inner_buf_size_ +=
        max_batch_size_ * seq_len_ * to_hidden_dim_ * sizeof(DataType_);
    return bw_inner_buf_size_;
  }

  void initialize(LinearTransposeParam<OpType> param) { param_ = param; }

  void forward(LinearTransposeForwardParam fw_param);
  void backward(LinearTransposeBackwardParam bw_param);

  ~LinearTranspose() {}
};
} // namespace fastertransformerv4