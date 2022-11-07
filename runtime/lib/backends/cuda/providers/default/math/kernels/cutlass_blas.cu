//===- cutlass_blas.cu ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "cutlass/cutlass.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "cutlass/gemm/device/gemm_batched.h"
#pragma GCC diagnostic pop
#include "cutlass/layout/matrix.h"

namespace brt {
namespace cuda {
namespace kernel {

// cutlass batch matmul implementation
template <typename T>
cutlass::Status
cutlass_batch_matmul(const T *A, int lda, long long int batch_stride_A,
                     const T *B, int ldb, long long int batch_stride_B, T *C,
                     int ldc, long long int batch_stride_C, int batch_count,
                     int m, int n, int k, T alpha, T beta,
                     cudaStream_t stream = nullptr) {
  using Gemm = cutlass::gemm::device::GemmBatched<T, cutlass::layout::RowMajor,
                                                  T, cutlass::layout::RowMajor,
                                                  T, cutlass::layout::RowMajor>;
  Gemm gemm_op;
  return gemm_op({{m, n, k},
                  {A, lda},
                  batch_stride_A,
                  {B, ldb},
                  batch_stride_B,
                  {C, ldc},
                  batch_stride_C,
                  {C, ldc},
                  batch_stride_C,
                  {alpha, beta},
                  batch_count},
                 nullptr, stream);
}

// instantiate
template cutlass::Status
cutlass_batch_matmul<float>(const float *, int, long long int, const float *,
                            int, long long int, float *, int, long long int,
                            int, int, int, int, float, float, cudaStream_t);

} // namespace kernel
} // namespace cuda
} // namespace brt
