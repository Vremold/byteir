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

SmallVector<bool> DimFlagAnalysis::GetDimFlag(Value value) {
  auto found = memorized_.find(value);

  if (found != memorized_.end()) {
    return found->second;
  }

  SmallVector<bool> res = compute_flag_->Compute(value);

  llvm::sys::SmartScopedLock<true> guard(mutex_);
  memorized_[value] = res;
  return res;
}
