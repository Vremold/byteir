//===- TilingUtils.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/TilingUtils.h"
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
void mlir::unpackRanges(ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                        SmallVectorImpl<Value> &ubs,
                        SmallVectorImpl<Value> &steps) {

  for (Range range : ranges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }
}

LogicalResult mlir::buildSCFLoop(
    OpBuilder &b, Location loc, bool isParallel, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  if (isParallel) {
    b.create<scf::ParallelOp>(loc, lbs.take_front(), ubs.take_front(),
                              steps.take_front(), bodyBuilder);
  } else {
    buildLoopNest(b, loc, lbs.take_front(), ubs.take_front(),
                  steps.take_front(), bodyBuilder);
  }

  return success();
}

LogicalResult mlir::buildAffineLoop(
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  llvm::SmallVector<int64_t, 4> stepLiterals;
  for (auto step : steps.take_front()) {
    auto lit = getLiteralFromConstantLike(step, -1);
    if (lit == -1) {
      return failure();
    }
    stepLiterals.push_back(lit);
  }

  buildAffineLoopNest(b, loc, lbs.take_front(), ubs.take_front(), stepLiterals,
                      bodyBuilder);

  return success();
}

Optional<linalg::LinalgOp> mlir::createAtomicLinalgGeneric(
    OpBuilder &b, Location loc, arith::AtomicRMWKind kind,
    ArrayRef<Value> inputs, ArrayRef<Value> outputs) {
  auto ctx = b.getContext();
  size_t num = inputs.size();

  // FIXME: only support all Ranks are equal now
  auto maybeRank = getRank(inputs.back());
  if (!maybeRank.hasValue())
    return llvm::None;
  auto rank = maybeRank.getValue();

  for (auto val : inputs) {
    auto anotherMaybeRank = getRank(val);
    if (!anotherMaybeRank.hasValue() || rank != anotherMaybeRank.getValue()) {
      return llvm::None;
    }
  }

  for (auto val : outputs) {
    auto anotherMaybeRank = getRank(val);
    if (!anotherMaybeRank.hasValue() || rank != anotherMaybeRank.getValue()) {
      return llvm::None;
    }
  }

  SmallVector<AffineMap, 2> indexingMaps;

  SmallVector<StringRef> parallelLoopAttrs(rank, getParallelIteratorTypeName());

  // insert identity map for input
  for (size_t i = 0; i < num; ++i) {
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  }

  // insert identity map for output
  for (size_t i = 0; i < num; ++i) {
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  }

  ValueRange inputRange(inputs);
  ValueRange outputRange(outputs);

  linalg::LinalgOp linalgOp = b.create<linalg::GenericOp>(
      loc, inputRange, outputRange, indexingMaps, parallelLoopAttrs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        SmallVector<Value, 2> indices;
        SmallVector<Value, 2> opResults;
        // create indices
        for (unsigned i = 0; i < rank; ++i) {
          auto index = nestedBuilder.create<linalg::IndexOp>(loc, i);
          indices.push_back(index.getResult());
        }

        // create
        for (size_t i = 0; i < num; ++i) {
          auto op = nestedBuilder.create<memref::AtomicRMWOp>(
              loc, blockArgs[i].getType(), kind, blockArgs[i], outputs[i],
              indices);

          opResults.push_back(op.getResult());
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResults);
      });

  linalgOp->setAttr(
      getAtomicKindAttrName(),
      IntegerAttr::get(IntegerType::get(ctx, 32), static_cast<int64_t>(kind)));

  return linalgOp;
}
