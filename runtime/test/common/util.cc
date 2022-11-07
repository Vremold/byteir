//===- util.cc ------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "brt/test/common/util.h"

namespace brt {
namespace test {

int64_t LinearizedShape(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto d : shape) {
    res *= d;
  }
  return res;
}

} // namespace test
} // namespace brt
