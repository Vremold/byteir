//===- DimFromBroadcast.h -------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H
#define BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H

#include "byteir/Analysis/DimFlag.h"

using namespace llvm;
using namespace mlir;

namespace byteir {

class DimFromBroadcast : public ComputeFlag {
  SmallVector<bool> Compute(Value v) override;
};
}

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_DIMFROMBROADCAST_H
