//===- ReduceFusion.cpp ----------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct PadReduceWindowPattern : public OpRewritePattern<mhlo::ReduceWindowOp> {
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {

    // avoid already fused
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    // only support cases of all pads or none pads
    size_t numPad = llvm::count_if(op.getInputs(), [&](Value v) {
      return isa_and_nonnull<mhlo::PadOp>(v.getDefiningOp());
    });

    MhloFusionPattern pattern;
    // handle the case of all pads
    if (numPad == op.getInputs().size()) {
      for (auto val : op.getInputs()) {
        auto pad = cast<mhlo::PadOp>(val.getDefiningOp());
        // handle pad of constant
        auto paddingValDefOp = pad.getPaddingValue().getDefiningOp();
        if (isSplatMhloConstant(paddingValDefOp)) {
          auto cloned = replicateDefiningOp(rewriter, pad, 1, 0);
          pattern.push_back(cloned);
        }

        pattern.push_back(pad);
      }
    } else {
      return failure();
    }

    // handle initial as a constant
    size_t idx = op.getInputs().size();
    for (auto val : op.getInitValues()) {
      auto initialDefOp = val.getDefiningOp();
      if (isSplatMhloConstant(initialDefOp)) {
        auto cloned = replicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
      }
      idx++;
    }

    pattern.push_back(op);

    auto fusion = createMhloFusionFromPattern(rewriter, pattern);

    // add attr
    fusion->setAttr(getByteIRReduceFusionAttrName(),
                    UnitAttr::get(fusion.getContext()));

    return success();
  }
};

struct ReduceFusionPass : public ReduceFusionBase<ReduceFusionPass> {

  ReduceFusionPass() : ReduceFusionBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    RewritePatternSet patterns(funcOp.getContext());
    populateFuseReduceWindowPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "ReduceFusionPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateFuseReduceWindowPatterns(RewritePatternSet &patterns) {
  patterns.add<PadReduceWindowPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createReduceFusionPass() {
  return std::make_unique<ReduceFusionPass>();
}