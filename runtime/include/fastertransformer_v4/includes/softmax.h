/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct SoftmaxForwardParam {
  const T *input;
  T *softmax_output;
  void *buf;
  const int rows;
  const int cols;
  cudaStream_t stream;
  bool add_mask = false;
  T *mask = NULL;
  int head_num = 1;
  bool apply_dropout = false;
  float dropout_rate = 0.0f;
  uint8_t *dropout_mask = NULL;
  T *softmax_dropout_output = NULL;
  bool batch_first = true; // for deberta
};

template <typename T> struct SoftmaxBackwardParam {
  const T *grad_out;
  const T *out;
  T *grad_in;
  void *buf;
  const int rows;
  const int cols;
  cudaStream_t stream;
  bool apply_dropout = false;
  float dropout_rate = 0.0f;
  uint8_t *dropout_mask = NULL;
};

template <OperationType OpType> class Softmax {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using SoftmaxForwardParam = struct SoftmaxForwardParam<DataType_>;
  using SoftmaxBackwardParam = struct SoftmaxBackwardParam<DataType_>;

public:
  Softmax() {}

  static void forward(SoftmaxForwardParam param);
  static void backward(SoftmaxBackwardParam param);

  ~Softmax() {}
};
} // namespace fastertransformerv4