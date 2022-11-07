//===- index_select.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cuda_runtime.h"
#include <cstdint>

namespace brt {
namespace cuda {
namespace kernel {
template <typename T>
void index_select(const T *input, const uint32_t *index, T *output, const int A,
                  const int IB, const int OB, const int C, cudaStream_t stream);
} // namespace kernel
} // namespace cuda
} // namespace brt