//===- kernels.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace brt {
namespace cuda {
namespace external_kernels {

// declaration

template <typename T>
void add_kernel(const T *input_1, const T *input_2, T *output, int n);

} // namespace external_kernels
} // namespace cuda
} // namespace brt
