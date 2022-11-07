//===- matmul.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {

/**
 * Matmul Ops
 */
template <typename T> class MatmulImpl {
public:
  explicit MatmulImpl(const OpAccessor &accessor);

  void Execute(const T *a_val, const T *b_val, T *c_val, cublasHandle_t handle,
               cudaStream_t stream);

private:
  bool lhs_transpose = false, rhs_transpose = false, output_transpose = false;
  int m, n, k;
  bool compute_on_fp16 = false;
  float alpha = 1.0f, beta = 0.0f;
};

template <typename T>
using Matmul = CublasOpKernel<MatmulImpl<T>, TypedOperand<const T *, 0>,
                              TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt
