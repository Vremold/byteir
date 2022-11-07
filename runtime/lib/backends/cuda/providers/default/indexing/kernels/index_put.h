//===- index_put.h --------------------------------------------*--- C++ -*-===//
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
template <typename T, bool Accum>
void index_put(const T *input, const int64_t *indices, const T *update,
               T *output, const int index_count, const int feature_bound,
               const int size, cudaStream_t stream);
} // namespace kernel
} // namespace cuda
} // namespace brt