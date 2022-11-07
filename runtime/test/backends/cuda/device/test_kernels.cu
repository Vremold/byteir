//===- test_kernels.cu ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

namespace brt {
namespace test {
// TODO move this kernel to another separate file
__global__ void test_kernel(const float *input, float *output, int n,
                            float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = input[i] + val;
  }
}
} // namespace test
} // namespace brt
