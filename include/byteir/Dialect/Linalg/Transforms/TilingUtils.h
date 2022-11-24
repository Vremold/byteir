//===- TilingUtils.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

constexpr StringRef getAtomicKindAttrName() { return "__byteir_atomic_kind__"; }

struct TileScope {
  Operation *anchorOp;
  llvm::SmallVector<Operation *> ops;

  TileScope() : anchorOp(nullptr) {}
  explicit TileScope(Operation *op) : anchorOp(op) {}
};

// TODO maybe relax this
inline bool isStructuralLinalg(mlir::linalg::LinalgOp op) {
  return op.getNumDpsInits() == 1;
}

void unpackRanges(OpBuilder &builder, Location loc, ArrayRef<Range> ranges,
                  SmallVectorImpl<Value> &lbs, SmallVectorImpl<Value> &ubs,
                  SmallVectorImpl<Value> &steps);

LogicalResult buildSCFLoop(OpBuilder &builder, Location loc, bool isParallel,
                           ValueRange lbs, ValueRange ubs, ValueRange steps,
                           function_ref<void(OpBuilder &, Location, ValueRange)>
                               bodyBuilder = nullptr);

// buildAffineLoop doesn't handle isParallel directly.
// Call affineParallelize after tiling instread.
LogicalResult buildAffineLoop(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder =
        nullptr);

// Create atomic add
llvm::Optional<linalg::LinalgOp>
createAtomicLinalgGeneric(OpBuilder &b, Location loc, arith::AtomicRMWKind kind,
                          ArrayRef<Value> inputs, ArrayRef<Value> outputs);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
