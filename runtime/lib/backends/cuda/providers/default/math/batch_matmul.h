//===- batch_matmul.h -----------------------------------------*--- C++ -*-===//
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
 * BatchMatmul Ops
 */
template <typename T> class BatchMatmulImpl {
public:
  explicit BatchMatmulImpl(const OpAccessor &accessor);

  void Execute(const T *a_val, const T *b_val, T *c_val, cudaStream_t stream);

private:
  int m, n, k, batch_count;
  long long int batch_stride_A, batch_stride_B, batch_stride_C;
  float alpha = 1.0f, beta = 0.0f;
};

template <typename T>
using BatchMatmul =
    CudaOpKernel<BatchMatmulImpl<T>, TypedOperand<const T *, 0>,
                 TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt
