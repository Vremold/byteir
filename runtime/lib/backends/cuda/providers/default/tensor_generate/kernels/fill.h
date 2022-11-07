//===- fill.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <cstdint>

namespace brt {
namespace cuda {
namespace kernel {
template <typename T>
void Fill(cudaStream_t stream, T *output, T value, size_t count);
} // namespace kernel
} // namespace cuda
} // namespace brt