//===- dtype.h ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once
#include "brt/core/framework/dtype.h"
#include <cuda_fp16.h>

namespace brt {

template <> struct ctype_to_dtype<__half> {
  static constexpr DTypeEnum value = DTypeEnum::Float16;
};

} // namespace brt
