//===- fill.cu ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "./fill.h"

// TODO: move to common header
#define DIVUP(x, y) (((x) + (y)-1) / (y))

namespace brt {
namespace cuda {
namespace kernel {
template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(T *output_data, T val, int32_t N) {
  int32_t id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T *output, T value, size_t count) {
  constexpr int maxThreadsPerBlock = 256;
  constexpr int maxElementsPerThread = 4;
  int blocksPerGrid =
      static_cast<int>(DIVUP(count, maxThreadsPerBlock * maxElementsPerThread));
  int32_t N = static_cast<int32_t>(count);
  _Fill<T, maxThreadsPerBlock, maxElementsPerThread>
      <<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(output, value, N);
}

#define INST(T) template void Fill<T>(cudaStream_t, T *, T, size_t);

INST(float)
INST(int64_t)
INST(double)
INST(__half)

#undef INST

} // namespace kernel
} // namespace cuda
} // namespace brt