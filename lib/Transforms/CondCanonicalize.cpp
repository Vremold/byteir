//===- CondCanonicalize.cpp --------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CondCanonicalize.h"
#include "./PassDetail.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::scf;

namespace {

template <typename OpTy> struct RemOfArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();

    // TODO: add support for if
    if (!isa_and_nonnull<LoopLikeOpInterface>(parentOp)) {
      return failure();
    }

    // paraentOp is a looplike
    auto looklike = cast<LoopLikeOpInterface>(parentOp);
    auto iv = getInductionVar(looklike);

    // FIXME handle negative cases
    if (op->getOperand(0) == iv &&
        confirmGEUpperBound(op->getOperand(1), looklike)) {
      rewriter.replaceOp(op, iv);
      return success();
    }

    return failure();
  };
};

template <typename OpTy> struct DivOfArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();

    // TODO: add support for if
    if (!isa_and_nonnull<LoopLikeOpInterface>(parentOp)) {
      return failure();
    }

    // paraentOp is a looplike
    auto looklike = cast<LoopLikeOpInterface>(parentOp);
    auto iv = getInductionVar(looklike);

    // FIXME handle negative cases
    if (op->getOperand(0) == iv &&
        confirmGEUpperBound(op->getOperand(1), looklike)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      return success();
    }

    return failure();
  };
};

struct CondCanonicalizePass
    : public CondCanonicalizeBase<CondCanonicalizePass> {
  CondCanonicalizePass() : CondCanonicalizeBase() {}

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    populateCondCanonicalizePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("CondCanonicalizePass applyPatternsAndFoldGreedily does "
                       "not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateCondCanonicalizePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  patterns
      .insert<DivOfArgFolder<arith::DivSIOp>, DivOfArgFolder<arith::DivUIOp>,
              RemOfArgFolder<arith::RemSIOp>, RemOfArgFolder<arith::RemUIOp>>(
          ctx);

  // add populateSCFForLoopCanonicalizationPatterns by default
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createCondCanonicalizePass() {
  return std::make_unique<CondCanonicalizePass>();
}
