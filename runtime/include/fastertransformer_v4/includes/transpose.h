/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct TransposeParam {
  const T *input;
  T *output;
  void *buf;
  const int dim_1;
  const int dim_2;
  const int dim_3;
  const int dim_4;
  cudaStream_t stream;
  transposeType transpose_type = TRANSPOSE0213;
};

template <OperationType OpType> class Transpose {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using TransposeParam = struct TransposeParam<DataType_>;

public:
  Transpose() {}

  static void forward(TransposeParam param);

  ~Transpose() {}
};
} // namespace fastertransformerv4