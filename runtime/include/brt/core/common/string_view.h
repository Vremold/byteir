//===- string_view.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#if ((__cplusplus >= 201703L) && !defined(__CUDA_ARCH__))
#include <string_view>
#endif

namespace brt {
#if ((__cplusplus >= 201703L) && !defined(__CUDA_ARCH__))
using StringView = std::string_view;
#else
class StringHandle;
using StringView = StringHandle *;
#endif
} // namespace brt
