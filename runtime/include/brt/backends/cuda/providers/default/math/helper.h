//===- helper.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"
#include <cudnn.h>
#include <string>
#include <vector>

namespace brt {
namespace cuda {

namespace conv {
void handleConvParam(const OpAccessor &accessor, const Shape &shape_input,
                     const Shape &shape_filter, const Shape &shape_output,
                     int64_t &N, int64_t &iC, int64_t &iH, int64_t &iW,
                     int64_t &oC, int64_t &oH, int64_t &oW, int64_t &kH,
                     int64_t &kW, int64_t &strideH, int64_t &strideW,
                     int64_t &paddingH, int64_t &paddingW, int64_t &dilateH,
                     int64_t &dilateW, cudnnTensorFormat_t &format);
} // namespace conv

} // namespace cuda
} // namespace brt
