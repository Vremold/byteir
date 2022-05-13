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
      auto fp_type = type.getElementType().template dyn_cast<FloatType>();
      assert(fp_type);
      Value zero = rewriter.create<mhlo::ConstOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fp_type.getFloatSemantics())));
      op->setOperand(2, zero);
    }
    if (!isSplatMhloConstant(variance)) {
      auto type = op.variance().getType().template cast<RankedTensorType>();
      auto fp_type = type.getElementType().template dyn_cast<FloatType>();
      assert(fp_type);
      Value zero = rewriter.create<mhlo::ConstOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fp_type.getFloatSemantics())));
      op->setOperand(3, zero);
    }
    return success();
  }
};

struct RewriteWithConstraintPass
    : RewriteWithConstraintBase<RewriteWithConstraintPass> {
  RewriteWithConstraintPass() = default;
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateRewriteWithConstraintConstraintPattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
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

std::unique_ptr<OperationPass<FuncOp>> mlir::createRewriteWithConstraintPass() {
  return std::make_unique<RewriteWithConstraintPass>();
}