//===- kernels.cu ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

namespace brt {
namespace cuda {
namespace external_kernels {

template <typename T>
__global__ void add_kernel(const T *input_1, const T *input_2, T *output,
                           int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    output[idx] = input_1[idx] + input_2[idx];
  }
}

// instantiate
template __global__ void add_kernel<float>(const float *, const float *,
                                           float *, int);
template __global__ void add_kernel<int>(const int *, const int *, int *, int);

} // namespace external_kernels
} // namespace cuda
} // namespace brt
