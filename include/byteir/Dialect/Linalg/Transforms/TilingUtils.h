//===- TilingUtils.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"

namespace mlir {

struct TileOpProperty {
  mlir::linalg::LinalgOp op;
  unsigned axis;
  unsigned rank;
};

struct TileScope {
  mlir::linalg::LinalgOp anchorOp;
  // store {op, axis, rank}
  llvm::SmallVector<TileOpProperty> tileOps;

  TileScope(mlir::linalg::LinalgOp op)
    : anchorOp(op) {}
};

// TODO maybe relax this
inline bool isStructuralLinalg(mlir::linalg::LinalgOp op) {
  return op.getNumOutputs() == 1;
}

void unpackRanges(
  ArrayRef<Range> ranges, 
  SmallVectorImpl<Value>& lbs,
  SmallVectorImpl<Value>& ubs,
  SmallVectorImpl<Value>& steps);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
