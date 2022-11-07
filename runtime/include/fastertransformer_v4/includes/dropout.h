/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct DropoutForwardParam {
  const T *dropout_in;
  uchar4 *dropout_mask;
  T *dropout_out;
  void *buf;
  const int N;
  cudaStream_t stream;
};

template <typename T> struct DropoutBackwardParam {
  const T *grad_out;
  const uchar4 *dropout_mask;
  T *grad_in;
  void *buf;
  const int N;
  cudaStream_t stream;
};

template <OperationType OpType> class Dropout {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using DropoutForwardParam = struct DropoutForwardParam<DataType_>;
  using DropoutBackwardParam = struct DropoutBackwardParam<DataType_>;

  const int max_N_;
  const float ratio_;
  unsigned long long inner_buf_size_;

public:
  Dropout(const int N, const float ratio) : max_N_(N), ratio_(ratio) {}

  unsigned long long cal_bufsize() {
    // inner_buf_size_ = 0;
    return 0;
  }

  void forward(DropoutForwardParam param);

  void backward(DropoutBackwardParam param);

  ~Dropout() {}
};
} // namespace fastertransformerv4