//===- rng.h --------------------------------------------------*--- C++ -*-===//
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

void RngUniform(cudaStream_t stream, float *ptr, size_t length, float low,
                float high);

} // namespace kernel
} // namespace cuda
} // namespace brt