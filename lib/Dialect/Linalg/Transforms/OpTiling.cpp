//===- OpTiling.cpp - Implementation of linalg Tiling --------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
// Some code from Tiling.cpp of IREE
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/Tiling.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PassDetail.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-op-tiling"

//===----------------------------------------------------------------------===//
// local utils
//===----------------------------------------------------------------------===//

/// Returns failure if the options are unsupported.
static LogicalResult
verifySupportedTilingOptions(PatternRewriter &rewriter, Operation *op,
                             const linalg::LinalgTilingOptions &options) {
  if (!options.interchangeVector.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported interchange during tiling");
  }
  if (options.loopType != linalg::LinalgTilingLoopType::Loops) {
    return rewriter.notifyMatchFailure(op,
                                       "only tiling with scf.for is supported");
  }
  return success();
}

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<arith::ConstantIndexOp>(
        loc, attr.cast<IntegerAttr>().getInt());
  }
  return valueOrAttr.get<Value>();
}

/// Returns true if loop is untiled. Only checks if the value is statically
/// zero. It is assumed that a `Value` defined by a constant op is already
/// converted to an `IntegerAttr` of that value. So here just return true if
/// this is an attribute with a zero value.
static bool isUntiledLoop(OpFoldResult valueOrAttr) {
  Optional<int64_t> intVal = getConstantIntValue(valueOrAttr);
  return intVal && *intVal == 0;
}

// TODO move to util
// Clone a TilingInterface and replace its tensor outputs with `replacements`.
// Note if outputs are buffers, it returns the original TilingInterface.
static TilingInterface cloneTilingInterfaceAndReplaceTensorOutputs(
    OpBuilder &builder, TilingInterface tilableOp, ValueRange replacements,
    bool isBuffer) {
  if (isBuffer) {
    return tilableOp;
  }

  BlockAndValueMapping bvm;
  ValueRange opOutputs =
      tilableOp->getOperands().take_back(tilableOp->getNumResults());

  for (const auto valPair : llvm::zip(opOutputs, replacements)) {
    bvm.map(std::get<0>(valPair), std::get<1>(valPair));
  }

  Operation *clonedOp = builder.clone(*tilableOp.getOperation(), bvm);
  return cast<TilingInterface>(clonedOp);
}

// TODO: move some code to public

/// Generates the tiled loops and the body by invoking the interface methods of
/// TiledOpInterface.
/// - `outputs` are the operands to use for outputs of the tiled operation.
/// - `tileSizes` are tile sizes specified for all loops of the operation. If a
///   loop is to be untiled it is set to 0.
/// - `iteratorType` is the type of the loop iterator returned by the
///   TiledOpInterface.
/// - `loopBounds` are the bounds of all the loops of the op returned by the
///   TiledOpInterface.
/// - `loopDepth` is the current loop depth being processed.
/// - `offsets` are the `Value`s that represent the position of the tile being
///   operated on. The offsets are computed as the tiled loops are being
///   generated.
/// - `distributionInfo` is the proc_id and nprocs `Value`s to be used for
///   distributed loops. It is a stack, and once an entry at the top of the
///   stack is used for distribution it is popped before processing the inner
///   loops.
static FailureOr<TiledOp>
tileInterfaceOpImpl(OpBuilder &builder, TilingInterface tilableOp,
                    ValueRange outputs, MutableArrayRef<OpFoldResult> tileSizes,
                    ArrayRef<utils::IteratorType> iteratorTypes,
                    ArrayRef<Range> loopBounds, unsigned loopDepth,
                    SmallVectorImpl<OpFoldResult> &offsets,
                    ArrayRef<linalg::ProcInfo> distributionInfo) {
  Location loc = tilableOp.getLoc();
  bool isBufferTiling = tilableOp->getNumResults() == 0;
  // If this is the innermost loop, then generated the tiled implementation of
  // the op by invoking the TiledOpInterface methods.
  if (loopDepth == tileSizes.size()) {
    TiledOp ret;

    auto cloned = cloneTilingInterfaceAndReplaceTensorOutputs(
        builder, tilableOp, outputs, isBufferTiling);
    SmallVector<Operation *> tiledOps =
        cloned.getTiledImplementation(builder, offsets, tileSizes);

    if (tiledOps.empty()) {
      return static_cast<LogicalResult>(
          tilableOp.emitOpError("failed to get tiled implementation"));
    }
    assert(
        tiledOps.size() == 1 &&
        "expected only a single operation returned from tiling implementation");
    ret.op = tiledOps[0];

    for (auto result : llvm::enumerate(ret.op->getResults())) {
      if (!result.value().getType().isa<RankedTensorType>()) {
        ret.results.push_back(result.value());
        continue;
      }

      SmallVector<OpFoldResult> resultOffsets, resultSizes;
      if (succeeded(tilableOp.getResultTilePosition(
              builder, result.index(), offsets, tileSizes, resultOffsets,
              resultSizes))) {
        SmallVector<OpFoldResult> resultStrides(resultOffsets.size(),
                                                builder.getIndexAttr(1));
        Value insertSlice = builder.create<tensor::InsertSliceOp>(
            loc, ret.op->getResult(result.index()), outputs[result.index()],
            resultOffsets, resultSizes, resultStrides);
        ret.results.push_back(insertSlice);
      }
    }
    return ret;
  }

  // If tile size at this depth is empty, do nothing.
  if (isUntiledLoop(tileSizes[loopDepth])) {
    auto zeroAttr = builder.getI64IntegerAttr(0);
    offsets.push_back(zeroAttr);
    tileSizes[loopDepth] = loopBounds[loopDepth].size;
    return tileInterfaceOpImpl(builder, tilableOp, outputs, tileSizes,
                               iteratorTypes, loopBounds, loopDepth + 1,
                               offsets, distributionInfo);
  }

  // Generate an scf.for for the current loop depth.
  Value lb = getValueOrCreateConstantIndexOp(builder, loc,
                                             loopBounds[loopDepth].offset);
  Value ub =
      getValueOrCreateConstantIndexOp(builder, loc, loopBounds[loopDepth].size);

  Value step = getValue(builder, loc, tileSizes[loopDepth]);

  // Update lb, ub and step for cyclic distribution.
  if (!distributionInfo.empty() &&
      iteratorTypes[loopDepth] == utils::IteratorType::parallel) {
    linalg::updateBoundsForCyclicDistribution(
        builder, loc, distributionInfo.front().procId,
        distributionInfo.front().nprocs, lb, ub, step);
    distributionInfo = distributionInfo.drop_front();
  }
  FailureOr<TiledOp> innerReturnValue;

  ValueRange initValues(isBufferTiling ? ValueRange{} : outputs);

  auto forOp = builder.create<scf::ForOp>(
      loc, lb, ub, step, initValues,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        offsets.push_back(iv);
        auto affineMaps = AffineMap::inferFromExprList({ArrayRef<AffineExpr>{
            b.getAffineSymbolExpr(0),
            b.getAffineSymbolExpr(1) - b.getAffineDimExpr(0)}})[0];
        // Similar to linalg tiling, the tile size is the min(tileSizes, ub -
        // iv) to account for cases where tile size does not divide (ub - lb)
        // exactly.
        Value inBoundsTileSize = b.create<AffineMinOp>(
            loc, affineMaps,
            ValueRange{iv, getValue(builder, loc, tileSizes[loopDepth]), ub});
        tileSizes[loopDepth] = getAsOpFoldResult(inBoundsTileSize);
        // Recursively proceed to generate the tiled loop for the next level.
        innerReturnValue =
            tileInterfaceOpImpl(b, tilableOp, (isBufferTiling ? outputs : args),
                                tileSizes, iteratorTypes, loopBounds,
                                loopDepth + 1, offsets, distributionInfo);

        if (failed(innerReturnValue))
          return;

        b.create<scf::YieldOp>(loc, innerReturnValue->results);
      });

  // add parallel annotation
  if (iteratorTypes[loopDepth] == utils::IteratorType::parallel)
    forOp->setAttr(getSCFForParallelAttrName(), builder.getUnitAttr());

  if (failed(innerReturnValue)) {
    return innerReturnValue;
  }
  innerReturnValue->loops.insert(innerReturnValue->loops.begin(),
                                 forOp.getOperation());
  innerReturnValue->results = forOp.getResults();
  return innerReturnValue;
}

FailureOr<TiledOp> tileInterfaceOp(OpBuilder &b, TilingInterface tilableOp,
                                   const linalg::LinalgTilingOptions &options) {
  // Gather destination tensors.
  SmallVector<Value> dest;
  Location loc = tilableOp.getLoc();

  if (failed(tensor::getOrCreateDestinations(b, loc, tilableOp, dest)))
    return tilableOp->emitOpError("failed to get destination tensors");

  SmallVector<utils::IteratorType> iteratorTypes =
      tilableOp.getLoopIteratorTypes();
  SmallVector<Value, 4> tileSizesVals =
      options.tileSizeComputationFunction(b, tilableOp);
  auto zeroAttr = b.getI64IntegerAttr(0);

  // The actual tile sizes used converts `Value` defined as constant 0, to a
  // zero integer attributes. Currently if the iterator type is not "parallel",
  // the tile size is forced to zero as well.
  // LWC FIXME to support reduction tiling
  auto tileSizes = getAsOpFoldResult(tileSizesVals);
  tileSizes.resize(iteratorTypes.size(), zeroAttr);
  for (auto en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == utils::IteratorType::parallel)
      continue;

// TODO change to checking a interface func later
#if 0
    if (!isUntiledLoop(tileSizes[en.index()])) {
      return static_cast<LogicalResult>(tilableOp.emitOpError(
          "unimplemented tiling of non-parallel loop iterator type"));
    }
#endif
  }

  // Trivial early exit case of tile sizes being zero for all parallel loops.
  if (llvm::all_of(tileSizes, isUntiledLoop)) {
    return TiledOp{tilableOp, {}, {}};
  }

  SmallVector<Range> loopBounds = tilableOp.getIterationDomain(b);
  SmallVector<linalg::ProcInfo> distributionInfo;
  // If the tiled loops are distributed, get the proc_id and nprocs for the
  // distributed loops. First collect the parallel loops by iterating over the
  // tileSizes and getting the loops that are distribute, i.e.,
  // - parallel, i.e. iteratorTypes is "parallel"
  // - tiled, i.e. tileSize != 0
  if (options.distribution) {
    SmallVector<Range> distributedLoopRange;
    for (auto i : llvm::seq<unsigned>(0, tileSizes.size())) {
      if (isUntiledLoop(tileSizes[i]))
        continue;
      if (iteratorTypes[i] != utils::IteratorType::parallel)
        continue;
      distributedLoopRange.push_back(loopBounds[i]);
    }
    distributionInfo = options.distribution->procInfo(b, tilableOp.getLoc(),
                                                      distributedLoopRange);
  }

  SmallVector<OpFoldResult> offsets;
  return tileInterfaceOpImpl(b, tilableOp, dest, tileSizes, iteratorTypes,
                             loopBounds, 0, offsets, distributionInfo);
}

FailureOr<TiledOp>
mlir::linalg_ext::tileLinalgExtOp(RewriterBase &rewriter,
                                  TilingInterface tilableOp,
                                  const linalg::LinalgTilingOptions &options) {
  // need to set point since tileInterfaceOp using builder
  rewriter.setInsertionPoint(tilableOp);
  FailureOr<TiledOp> tiledOp = tileInterfaceOp(rewriter, tilableOp, options);
  if (failed(tiledOp))
    return failure();

  if (tiledOp->op != tilableOp) {
    if (tiledOp->results.empty()) {
      rewriter.eraseOp(tilableOp);
    } else {
      rewriter.replaceOp(tilableOp, tiledOp->results);
    }
  }
  return tiledOp;
}

//////////////
// global
//////////////

LogicalResult
TilingInterfaceBaseTilingPattern::matchAndRewriteBase(TilingInterface tilableOp,
                                                      PatternRewriter &rewriter,
                                                      TiledOp &result) const {
  if (failed(filter.checkAndNotify(rewriter, tilableOp))) {
    return failure();
  }
  if (failed(verifySupportedTilingOptions(rewriter, tilableOp, options))) {
    return failure();
  }

  FailureOr<TiledOp> res = tileInterfaceOp(rewriter, tilableOp, options);
  if (failed(res))
    return res;
  result = *res;

  if (result.op) {
    filter.replaceLinalgTransformationFilter(rewriter, result.op);
  }
  return success();
}

LogicalResult
TilingInterfaceTilingPattern::matchAndRewrite(TilingInterface tilableOp,
                                              PatternRewriter &rewriter) const {
  // `LinalgOp`s also implement the `TilingInterface`. Do not handle LinalgOps
  // in this pattern. For now use these only for `LinalgExt` ops. This pattern
  // is to be deprecated to use something that can handle all `TilingInterface`
  // ops.
  if (isa<linalg::LinalgOp>(tilableOp.getOperation())) {
    return rewriter.notifyMatchFailure(tilableOp, "ignoring LinalgOps");
  }
  TiledOp tiledOp;
  // Check for failure.
  if (failed(TilingInterfaceBaseTilingPattern::matchAndRewriteBase(
          tilableOp, rewriter, tiledOp))) {
    return failure();
  }
  // Check for do-nothing case.
  if (!tiledOp.op)
    return failure();
  if (tiledOp.op != tilableOp) {
    if (tiledOp.results.empty()) {
      rewriter.eraseOp(tilableOp);
    } else {
      rewriter.replaceOp(tilableOp, tiledOp.results);
    }
  }
  return success();
}

namespace {
struct LinalgOpTilingPass : public LinalgOpTilingBase<LinalgOpTilingPass> {
  LinalgOpTilingPass() = default;
  LinalgOpTilingPass(ArrayRef<int64_t> tileSizes,
                     LinalgTilingLoopType loopType) {
    this->tileSizes = tileSizes;
    this->loopType = "";
    this->loopTypeEnum = loopType;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    RewritePatternSet patterns(context);

    patterns.add<TilingInterfaceTilingPattern>(
        context, linalg::LinalgTilingOptions().setTileSizes(tileSizes),
        linalg_ext::LinalgTransformationFilter(
            StringAttr::get(context, "outer_reduce_input"),
            StringAttr::get(context, "outer_reduce_output")));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LinalgTilingLoopType loopTypeEnum;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgOpTilingPass(ArrayRef<int64_t> tileSizes,
                               linalg::LinalgTilingLoopType loopType) {
  return std::make_unique<LinalgOpTilingPass>(tileSizes, loopType);
}
