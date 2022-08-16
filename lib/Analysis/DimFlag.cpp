//===- DimFlag.cpp --------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/DimFlag.h"

using namespace byteir;
using namespace llvm;
using namespace mlir;

SmallVector<bool> DimFlagAnalysis::getDimFlag(Value value) {
  auto found = memorized.find(value);

  if (found != memorized.end()) {
    return found->second;
  }

  SmallVector<bool> res = computeFlag->compute(value);

  llvm::sys::SmartScopedLock<true> guard(mutex);
  memorized[value] = res;
  return res;
}
