//===- value.h ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint> // include this header for int64_t
#include <vector>

namespace brt {
// Before implementing AsyncValue, we use "using"
// TODO: remove this after implementing AsyncValue
using AsyncValueRef = void *;
using AsyncValue = void *;
// Same as Shape
using ShapeRef = std::vector<int64_t> &;
using Shape = std::vector<int64_t>;

} // namespace brt
