//===- DimFromBroadcast.h -------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H
#define BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H

#include "byteir/Analysis/DimFlag.h"

namespace byteir {

class DimFromBroadcast : public ComputeFlag {
  llvm::SmallVector<bool> Compute(mlir::Value v) override;
};
} // namespace byteir

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H
