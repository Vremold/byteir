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

namespace byteir {

struct DimFlagAnalysis;

class ComputeFlag {
public:
  void setAnalysis(DimFlagAnalysis *analys) {
    assert(!analysis && "analysis is already set.");
    analysis = analys;
  }
  virtual llvm::SmallVector<bool> compute(mlir::Value v) = 0;

protected:
  DimFlagAnalysis *analysis{nullptr};
};

// `DimFlagAnalysis` is used to get the flag of each dim in a Value. Users need
// to implement a subclass of `ComputeFlag` to define how the flags will be
// computed.
struct DimFlagAnalysis {
  DimFlagAnalysis(ComputeFlag *flag) : computeFlag(flag) {
    computeFlag->setAnalysis(this);
  }
  llvm::SmallVector<bool> getDimFlag(mlir::Value value);

  llvm::DenseMap<mlir::Value, llvm::SmallVector<bool>> memorized;
  ComputeFlag *computeFlag;
  llvm::sys::SmartMutex<true> mutex;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_DIMFLAG_H
