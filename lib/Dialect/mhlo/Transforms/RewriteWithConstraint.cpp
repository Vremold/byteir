//===- RewriteWithConstraint.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/RewriteWithConstraint.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct BatchNormGradDropMeanAndVarPattern
    : public OpRewritePattern<mhlo::BatchNormGradOp> {
  using OpRewritePattern<mhlo::BatchNormGradOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormGradOp op,
                                PatternRewriter &rewriter) const override {
    auto mean = op.mean().getDefiningOp();
    auto variance = op.variance().getDefiningOp();
    if (isSplatMhloConstant(mean) && isSplatMhloConstant(variance)) {
      return failure();
    }
    if (!isSplatMhloConstant(mean)) {
      auto type = op.mean().getType().template cast<RankedTensorType>();
      auto fpType = type.getElementType().template dyn_cast<FloatType>();
      assert(fpType);
      Value zero = rewriter.create<mhlo::ConstantOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fpType.getFloatSemantics())));
      op->setOperand(2, zero);
    }
    if (!isSplatMhloConstant(variance)) {
      auto type = op.variance().getType().template cast<RankedTensorType>();
      auto fpType = type.getElementType().template dyn_cast<FloatType>();
      assert(fpType);
      Value zero = rewriter.create<mhlo::ConstantOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fpType.getFloatSemantics())));
      op->setOperand(3, zero);
    }
    return success();
  }
};

struct RewriteWithConstraintPass
    : RewriteWithConstraintBase<RewriteWithConstraintPass> {
  RewriteWithConstraintPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateRewriteWithConstraintConstraintPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("RewriteWithConstraintPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateRewriteWithConstraintConstraintPattern(
    RewritePatternSet &patterns) {
  patterns.add<BatchNormGradDropMeanAndVarPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRewriteWithConstraintPass() {
  return std::make_unique<RewriteWithConstraintPass>();
}