//===- DimFlag.h ----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_DIMFLAG_H
#define BYTEIR_ANALYSIS_DIMFLAG_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Mutex.h"

using namespace llvm;
using namespace mlir;

namespace byteir {

struct DimFlagAnalysis;

class ComputeFlag {
public:
  void SetAnalysis(DimFlagAnalysis *analysis) {
    assert(!analysis_ && "analysis_ is already set.");
    analysis_ = analysis;
  }
  virtual SmallVector<bool> Compute(Value v) = 0;

protected:
  DimFlagAnalysis *analysis_{nullptr};
};

// `DimFlagAnalysis` is used to get the flag of each dim in a Value. Users need
// to implement a subclass of `ComputeFlag` to define how the flags will be
// computed.
struct DimFlagAnalysis {
  DimFlagAnalysis(ComputeFlag *compute_flag) : compute_flag_(compute_flag) {
    compute_flag_->SetAnalysis(this);
  }
  SmallVector<bool> GetDimFlag(Value value);

  llvm::DenseMap<Value, SmallVector<bool>> memorized_;
  ComputeFlag *compute_flag_;
  sys::SmartMutex<true> mutex_;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_DIMFLAG_H
