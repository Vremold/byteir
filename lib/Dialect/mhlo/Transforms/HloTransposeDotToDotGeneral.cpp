//===- HloTransposeDotToDotGeneral.cpp ---------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloTransposeDotToDotGeneral.h"
#include "PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {

// mhlo.transpose + mhlo.dot -> mhlo.dot_general
struct FuseTransposeDotToDotGeneralPattern
    : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs_transpose = op.lhs().getDefiningOp<mhlo::TransposeOp>();
    auto rhs_transpose = op.rhs().getDefiningOp<mhlo::TransposeOp>();
    if (!lhs_transpose && !rhs_transpose) {
      return failure();
    }
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    int64_t lhs_contracting_dimension = 1;
    int64_t rhs_contracting_dimension = 0;
    if (lhs_transpose) {
      lhs_contracting_dimension = 0;
      lhs = lhs_transpose.operand();
    }
    if (rhs_transpose) {
      rhs_contracting_dimension = 1;
      rhs = rhs_transpose.operand();
    }
    auto dimension_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, {lhs_contracting_dimension},
        {rhs_contracting_dimension});
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, op.getResult().getType(), lhs, rhs, dimension_numbers,
        op.precision_configAttr());
    return success();
  }
};

struct HloTransposeDotToDotGeneralPass
    : public HloTransposeDotToDotGeneralBase<HloTransposeDotToDotGeneralPass> {
  HloTransposeDotToDotGeneralPass() = default;
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloTransposeDotToDotGeneralPattern(patterns);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloTransposeDotToDotGeneralPattern(
    RewritePatternSet &patterns) {
  patterns.add(std::make_unique<FuseTransposeDotToDotGeneralPattern>(
      patterns.getContext()));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHloTransposeDotToDotGeneralPass() {
  return std::make_unique<HloTransposeDotToDotGeneralPass>();
}