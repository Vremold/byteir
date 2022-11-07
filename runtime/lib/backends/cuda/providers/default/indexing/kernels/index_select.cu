//===- index_select.cu ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./index_select.h"
#include <algorithm>

namespace brt {
namespace cuda {
namespace kernel {

template <typename T>
__global__ void naive_index_select_kernel(const T *input, const uint32_t *index,
                                          T *output, const int A, const int IB,
                                          const int OB, const int C) {
  for (int outIdx = blockIdx.x * blockDim.x + threadIdx.x; outIdx < A * OB * C;
       outIdx += gridDim.x * blockDim.x) {
    const int ind = outIdx / C % OB;
    const int inpIdx =
        outIdx / (OB * C) * (IB * C) + index[ind] * C + outIdx % C;
    output[outIdx] = input[inpIdx];
  }
}

template <typename T>
void index_select(const T *input, const uint32_t *index, T *output, const int A,
                  const int IB, const int OB, const int C,
                  cudaStream_t stream) {
  dim3 grid = std::min(256, (A * OB * C + 63) / 64);
  dim3 block = std::min(64, A * OB * C);
  naive_index_select_kernel<<<grid, block, 0, stream>>>(input, index, output, A,
                                                        IB, OB, C);
}

template void index_select<float>(const float *, const uint32_t *, float *,
                                  const int, const int, const int, const int,
                                  cudaStream_t);

} // namespace kernel
} // namespace cuda
} // namespace brt
