//===- CondCanonicalize.cpp --------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CondCanonicalize.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

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

// Fold away ForOp iter arguments when having `__byteir_parallel__` attribute
struct ByteIRParallelForOpIterArgsFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {

    if (!forOp->hasAttr(getSCFForParallelAttrName())) {
      return failure();
    }

    if (llvm::all_of(forOp.getRegionIterArgs(),
                     [](Value val) { return val.use_empty(); })) {
      return failure();
    }

    // replace args with operands
    // meaning that it removes loop carry.
    // Later, iter arguments will be futher removed
    // by ForOpIterArgsFolder (in Canonicalizer)
    for (auto it : llvm::zip(forOp.getIterOperands(),  // iter from outside
                             forOp.getRegionIterArgs() // iter inside region
                             )) {
      std::get<1>(it).replaceAllUsesWith(std::get<0>(it));
    }

    return success();
  }
};

struct CondCanonicalizePass
    : public CondCanonicalizeBase<CondCanonicalizePass> {
  CondCanonicalizePass() : CondCanonicalizeBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    populateCondCanonicalizePatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("CondCanonicalizePass applyPatternsAndFoldGreedily does "
                       "not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateCondCanonicalizePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.add<ByteIRParallelForOpIterArgsFolder,
               DivOfArgFolder<arith::DivSIOp>, 
               DivOfArgFolder<arith::DivUIOp>,
               RemOfArgFolder<arith::RemSIOp>, 
               RemOfArgFolder<arith::RemUIOp>>(ctx);
  // clang-format on

  // add populateSCFForLoopCanonicalizationPatterns by default
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createCondCanonicalizePass() {
  return std::make_unique<CondCanonicalizePass>();
}
