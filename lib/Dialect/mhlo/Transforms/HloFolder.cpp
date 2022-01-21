//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "PassDetail.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {

static LogicalResult AddScatterAddMatchAndRewriteHelper(
  mhlo::AddOp add_op, int idx,
  PatternRewriter& rewriter) {

  // Match 
  mhlo::ScatterOp scatter_op =
    add_op.getOperand(idx).getDefiningOp<mhlo::ScatterOp>();

  if (!scatter_op) {
    return failure();
  }

  // check wthether scatter supported
  Region& region = scatter_op.update_computation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return failure();
  }

  auto& block = region.front();
  if (!IsBlockSingleAdd(&block)) {
    return failure();
  }

  Value initial_val = scatter_op.operand();
  if (!IsSplatMhloConstantValue(initial_val, (int64_t)0) &&
      !IsSplatMhloConstantValue(initial_val, 0.0)) {
    return failure();
  }

  // Rewrite
  int another_idx = 1 - idx;
  auto cloned = rewriter.clone(*scatter_op.getOperation());
  cloned->setOperand(0, add_op.getOperand(another_idx));
  rewriter.replaceOp(add_op, cloned->getResult(0));
  return success();
}

// Add + Scatter {add} -> Scatter
// TODO other scatter support
struct AddScatterAddToScatterPattern
  : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
    PatternRewriter& rewriter) const override {

    // handle left
    if (failed(AddScatterAddMatchAndRewriteHelper(op, 0, rewriter))) {
      // handle right
      return AddScatterAddMatchAndRewriteHelper(op, 1, rewriter);
    }
 
    return success();
  }
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloFoldPatterns(patterns);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  patterns.add<AddScatterAddToScatterPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}