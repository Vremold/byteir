//===- DotTransposeFusion.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/DotTransposeFusion.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

// mhlo.dot + mhlo.transpose -> mhlo.fusion
// mhlo.dot_general + mhlo.transpose -> mhlo.fusion
struct FuseDotTransposePattern : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    SmallVector<Value> inputs, outputs;
    MhloFusionPattern pattern;
    NamedAttrList attrs;
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("MatmulOp"));
    if (mhlo::DotOp dot = op.operand().getDefiningOp<mhlo::DotOp>()) {
      if (dot.lhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      if (dot.rhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      inputs.push_back(dot.lhs());
      inputs.push_back(dot.rhs());
      byre::appendByreComputeAttr(attrs, "output_transpose",
                                  rewriter.getUnitAttr());
      byre::appendByreComputeAttr(attrs, "lhs_contracting_dimension",
                                  rewriter.getI64IntegerAttr(1));
      byre::appendByreComputeAttr(attrs, "rhs_contracting_dimension",
                                  rewriter.getI64IntegerAttr(0));
      pattern.push_back(dot);
    } else if (mhlo::DotGeneralOp dot_general =
                   op.operand().getDefiningOp<mhlo::DotGeneralOp>()) {
      if (dot_general.lhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      if (dot_general.rhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      auto dot_dimension_numbers = dot_general.dot_dimension_numbers();
      if (dot_dimension_numbers.getLhsBatchingDimensions().size() != 0) {
        return failure();
      }
      if (dot_dimension_numbers.getRhsBatchingDimensions().size() != 0) {
        return failure();
      }
      if (dot_dimension_numbers.getLhsContractingDimensions().size() != 1) {
        return failure();
      }
      if (dot_dimension_numbers.getRhsContractingDimensions().size() != 1) {
        return failure();
      }
      inputs.push_back(dot_general.lhs());
      inputs.push_back(dot_general.rhs());
      byre::appendByreComputeAttr(attrs, "output_transpose",
                                  rewriter.getUnitAttr());
      byre::appendByreComputeAttr(
          attrs, "lhs_contracting_dimension",
          rewriter.getI64IntegerAttr(
              dot_dimension_numbers.getLhsContractingDimensions()[0]));
      byre::appendByreComputeAttr(
          attrs, "rhs_contracting_dimension",
          rewriter.getI64IntegerAttr(
              dot_dimension_numbers.getRhsContractingDimensions()[0]));
      pattern.push_back(dot_general);
    } else {
      return failure();
    }
    pattern.push_back(op);
    outputs.push_back(op.getResult());

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(getContext()));
    return success();
  }
};

struct DotTransposeFusionPass
    : public DotTransposeFusionBase<DotTransposeFusionPass> {
  DotTransposeFusionPass() = default;
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateDotTransposeFusionPattern(patterns);
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateDotTransposeFusionPattern(RewritePatternSet &patterns) {
  patterns.add(
      std::make_unique<FuseDotTransposePattern>(patterns.getContext()));
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createDotTransposeFusionPass() {
  return std::make_unique<DotTransposeFusionPass>();
}