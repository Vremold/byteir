//===- OpDependence.h -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_OPDEPENDENCE_H
#define BYTEIR_ANALYSIS_OPDEPENDENCE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

// declare
struct OpDependenceInfoImpl;

// handle OpDependenceInfo within a block
class OpDependenceInfo {
public:
  explicit OpDependenceInfo(Block *b);

  ~OpDependenceInfo();

  // opFrom properly depends opTo means opFrom and opTo has a connected path
  // from opFrom to opTo.
  // "Properly" means this function assumes OpFrom is not opTo
  bool properlyDepends(Operation *opFrom, Operation *opTo);

  // "depends" means either opFrom equal to opTo,
  // or opFrom properly depends opTo.
  bool depends(Operation *opFrom, Operation *opTo);

private:
  Block *block;

  std::unique_ptr<OpDependenceInfoImpl> impl;
};

} // namespace mlir

#endif // BYTEIR_ANALYSIS_OPDEPENDENCE_H