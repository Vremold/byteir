//===- elementwise.h ------------------------------------------*--- C++ -*-===//
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
void add_kernel(const T *input_1, const T *input_2, T *output, int n);

} // namespace kernel
} // namespace cuda
} // namespace brt
