//===- ClusterConstraint.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ClusterConstraint.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {
static inline bool matchConstantValueFloatOrSplat(mlir::Value value,
                                                  mlir::FloatAttr *attr) {
  if (matchPattern(value, m_Constant(attr))) {
    return true;
  }
  SplatElementsAttr denseAttr;
  if (matchPattern(value, m_Constant(&denseAttr))) {
    auto splatValue = denseAttr.getSplatValue<Attribute>();
    if (splatValue.isa<FloatAttr>()) {
      *attr = splatValue.cast<FloatAttr>();
      return true;
    }
  }
  return false;
}

static inline mhlo::FusionOp fuseWithConstantArgs(Operation *op,
                                                  std::vector<size_t> constArgs,
                                                  PatternRewriter &rewriter) {
  MhloFusionPattern pattern;
  // clone each constant arg since they might be shared with another operation
  for (auto &i : constArgs) {
    auto cloned = ReplicateDefiningOp(rewriter, op, i, 0);
    pattern.push_back(cloned);
  }
  pattern.push_back(op);
  return createMhloFusionFromPattern(rewriter, pattern);
}

struct RngUniformConstraint : public OpRewritePattern<mhlo::RngUniformOp> {
  using OpRewritePattern<mhlo::RngUniformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngUniformOp op,
                                PatternRewriter &rewriter) const override {
    // avoid already fused
    if (op->template getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    // check both low and high are constants
    FloatAttr low, high;
    if (!matchConstantValueFloatOrSplat(op->getOperand(0), &low) ||
        !matchConstantValueFloatOrSplat(op->getOperand(1), &high)) {
      return failure();
    }
    // ensure it has static-known shape
    DenseIntElementsAttr static_shape;
    if (!matchPattern(op->getOperand(2), m_Constant(&static_shape))) {
      return failure();
    }

    auto fusion = fuseWithConstantArgs(op, {0, 1, 2}, rewriter);
    fusion->setAttr(byre::getByreForceComputeNameAttrName(),
                    UnitAttr::get(fusion.getContext()));
    fusion->setAttr(byre::getByrePrefix() + "low", low);
    fusion->setAttr(byre::getByrePrefix() + "high", high);
    fusion->setAttr(byre::getByreComputeName(),
                    rewriter.getStringAttr("RngUniform"));
    return success();
  }
};

struct ClusterConstraintPass
    : public ClusterConstraintBase<ClusterConstraintPass> {
  ClusterConstraintPass() : ClusterConstraintBase() {}
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateClusterConstraintPattern(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("ClusterConstraintPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateClusterConstraintPattern(RewritePatternSet &patterns) {
  patterns.add<RngUniformConstraint>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createClusterConstraintPass() {
  return std::make_unique<ClusterConstraintPass>();
}
