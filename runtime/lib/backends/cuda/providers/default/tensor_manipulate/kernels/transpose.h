//===- transpose.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace brt {
namespace cuda {
namespace kernel {

// declaration
template <typename T>
void transpose_naive_2d(const T *input, T *output, int m, int n, dim3 grid,
                        dim3 block, cudaStream_t stream);

} // namespace kernel
} // namespace cuda
} // namespace brt
