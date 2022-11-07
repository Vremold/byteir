//===- cutlass_blas.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once
#include "cutlass/cutlass.h"
#include <cuda_runtime.h>

namespace brt {
namespace cuda {
namespace kernel {

// declaration

template <typename T>
cutlass::Status
cutlass_batch_matmul(const T *A, int lda, long long int batch_stride_A,
                     const T *B, int ldb, long long int batch_stride_B, T *C,
                     int ldc, long long int batch_stride_C, int batch_count,
                     int m, int n, int k, T alpha, T beta,
                     cudaStream_t stream = nullptr);

} // namespace kernel
} // namespace cuda
} // namespace brt
