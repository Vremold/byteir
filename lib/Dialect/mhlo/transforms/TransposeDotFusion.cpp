//===- TransposeDotFusion.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/transforms/TransposeDotFusion.h"
#include "PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {

struct FuseTransposeDotPattern : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto lhs_transpose =
        dyn_cast_or_null<mhlo::TransposeOp>(op.lhs().getDefiningOp());
    auto rhs_transpose =
        dyn_cast_or_null<mhlo::TransposeOp>(op.rhs().getDefiningOp());
    if (lhs_transpose || rhs_transpose) {
      int64_t lhs_contracting_dimension = lhs_transpose ? 0 : 1;
      int64_t rhs_contracting_dimension = rhs_transpose ? 1 : 0;
      Value lhs = lhs_transpose ? lhs_transpose.operand() : op.lhs();
      Value rhs = rhs_transpose ? rhs_transpose.operand() : op.rhs();

      Location loc =
          lhs_transpose
              ? rewriter.getFusedLoc({lhs_transpose->getLoc(), op->getLoc()})
              : op->getLoc();
      loc = rhs_transpose ? rewriter.getFusedLoc({rhs_transpose->getLoc(), loc})
                          : loc;
      mhlo::FusionOp fusionOp = rewriter.create<mhlo::FusionOp>(
          loc, op.getResult().getType(), ArrayRef<Value>{lhs, rhs});
      NamedAttrList attrs;
      // TODO: move this outside
      attrs.append(byre::getByreComputeName(),
                   rewriter.getStringAttr("MatmulOp"));
      byre::appendByreComputeAttr(
          attrs, "lhs_contracting_dimension",
          rewriter.getI64IntegerAttr(lhs_contracting_dimension));
      byre::appendByreComputeAttr(
          attrs, "rhs_contracting_dimension",
          rewriter.getI64IntegerAttr(rhs_contracting_dimension));
      fusionOp->setAttrs(attrs.getDictionary(getContext()));

      Region &region = fusionOp.fused_computation();
      Block &block = region.emplaceBlock();
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&block);
        Value new_dot_lhs = lhs_transpose ? rewriter.create<mhlo::TransposeOp>(
                                                loc, op.lhs().getType(), lhs,
                                                lhs_transpose.permutation())
                                          : lhs;
        Value new_dot_rhs = rhs_transpose ? rewriter.create<mhlo::TransposeOp>(
                                                loc, op.rhs().getType(), rhs,
                                                rhs_transpose.permutation())
                                          : rhs;
        Value dot = rewriter.create<mhlo::DotOp>(loc, op.getResult().getType(),
                                                 new_dot_lhs, new_dot_rhs,
                                                 op.precision_configAttr());
        rewriter.create<mhlo::ReturnOp>(loc, dot);
      }
      op->replaceAllUsesWith(fusionOp.getResults());
      return success();
    }
    return failure();
  }
};

struct TransposeDotFusionPass
    : public TransposeDotFusionBase<TransposeDotFusionPass> {
  TransposeDotFusionPass() = default;
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add(std::make_unique<FuseTransposeDotPattern>(context));
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<FunctionPass> mlir::createTransposeDotFusionPass() {
  return std::make_unique<TransposeDotFusionPass>();
}