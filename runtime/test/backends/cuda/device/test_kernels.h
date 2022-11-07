//===- test_kernels.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace brt {
namespace test {
void test_kernel(const float *input, float *output, int n, float val);
} // namespace test
} // namespace brt
