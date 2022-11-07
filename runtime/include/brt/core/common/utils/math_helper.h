//===- math_helper.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"
#include <string>
#include <vector>

namespace brt {

namespace matmul {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape,
                                       int64_t lhs_contracting_dimension,
                                       int64_t rhs_contracting_dimension,
                                       bool output_transpose);
} // namespace matmul

namespace batchmatmul {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape);
} // namespace batchmatmul

namespace conv {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &input_shape,
                                       const std::vector<int64_t> &filter_shape,
                                       const std::string &layout,
                                       int64_t strideH, int64_t strideW,
                                       int64_t paddingH, int64_t paddingW,
                                       int64_t dilateH, int64_t dilateW);
} // namespace conv

namespace pool {
std::vector<int64_t>
DeduceOutputShape(const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &window_dimensions,
                  const std::vector<int64_t> &window_strides,
                  const std::vector<int64_t> &padding);

void CalculatePitches(const std::vector<int64_t> &vec,
                      std::vector<int> &pitches);

size_t FindLeadingNonOnePositive(const std::vector<int64_t> &vec);
} // namespace pool

namespace reduction {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &src_shape,
                                       const std::vector<int64_t> &dimensions);
} // namespace reduction

namespace transpose {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &input_shape,
                                       const std::vector<int64_t> &permutation);
} // namespace transpose

} // namespace brt
