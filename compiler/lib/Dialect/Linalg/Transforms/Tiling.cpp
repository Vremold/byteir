﻿//===- Tiling.cpp - Implementation of linalg Tiling -----------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/Tiling.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-ext-tiling"

LogicalResult linalg_ext::TilingInterfaceBaseTilingPattern::matchAndRewriteBase(
    TilingInterface tilableOp, PatternRewriter &rewriter,
    scf::SCFTilingResult &result) const {
  if (failed(filter.checkAndNotify(rewriter, tilableOp))) {
    return failure();
  }

  FailureOr<scf::SCFTilingResult> res =
      tileUsingSCFForOp(rewriter, tilableOp, options);

  if (failed(res))
    return res;

  result = *res;

  if (result.tiledOp) {
    filter.replaceLinalgTransformationFilter(rewriter, result.tiledOp);
  }

  if (failed(isValidTiling(result.tiledOp))) {
    return tilableOp.emitOpError("has invalid tiling");
  }
  labelTileLoopType(result.tiledOp, result.loops);
  return success();
}

LogicalResult linalg_ext::TilingInterfaceTilingPattern::matchAndRewrite(
    TilingInterface tilableOp, PatternRewriter &rewriter) const {
  // `LinalgOp`s also implement the `TilingInterface`. Do not handle LinalgOps
  // in this pattern. For now use these only for `LinalgExt` ops. This pattern
  // is to be deprecated to use something that can handle all `TilingInterface`
  // ops.
  if (isa<linalg::LinalgOp>(tilableOp.getOperation())) {
    return rewriter.notifyMatchFailure(tilableOp, "ignoring LinalgOps");
  }
  scf::SCFTilingResult tiledOp;
  // Check for failure.
  if (failed(TilingInterfaceBaseTilingPattern::matchAndRewriteBase(
          tilableOp, rewriter, tiledOp))) {
    return failure();
  }
  // Check for do-nothing case.
  if (!tiledOp.tiledOp)
    return failure();
  if (tiledOp.tiledOp != tilableOp) {
    if (tiledOp.replacements.empty()) {
      rewriter.eraseOp(tilableOp);
    } else {
      rewriter.replaceOp(tilableOp, tiledOp.replacements);
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
        context, scf::SCFTilingOptions().setTileSizes(tileSizes),
        linalg_ext::LinalgTransformationFilter(
            StringAttr::get(context, getLinalgExtTileAttrName())));

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