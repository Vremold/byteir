/*
 * Author: Xiaoying Jia, Changyi Wan
 * Project: Faster Transformer Training
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying, wanchangyi}@bytedance.com
 */
#pragma once
#include "common.h"

namespace fastertransformerv4 {
template <typename T> struct MatMulForwardParam {
  const T *input_A;
  const T *input_B;
  T *output;
  const int M;
  const int K;
  const int N;
  const int batch_count = 1;
  const float scale = 1.0f;
  const bool A_T = false;
  const bool B_T = false;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <typename T> struct MatMulBackwardParam {
  const T *grad_out;
  const T *input_A;
  const T *input_B;
  T *grad_A;
  T *grad_B;
  const int M;
  const int K;
  const int N;
  const int batch_count = 1;
  const float scale = 1.0f;
  const bool A_T = false;
  const bool B_T = false;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType> class MatMul {
private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  using MatMulForwardParam = struct MatMulForwardParam<DataType_>;
  using MatMulBackwardParam = struct MatMulBackwardParam<DataType_>;

public:
  MatMul() {}

  static void forward(MatMulForwardParam fw_param); // not support broadcast
                                                    // yet, assert A.dim = B.dim
  static void backward(MatMulBackwardParam bw_param);

  ~MatMul() {}
};
} // namespace fastertransformerv4