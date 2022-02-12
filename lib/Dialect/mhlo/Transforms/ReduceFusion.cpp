//===- ReduceFusion.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ReduceFusion.h"
#include "PassDetail.h"
#include "byteir/Dialect/mhlo/Transforms/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct PadReduceWindowPattern : public OpRewritePattern<mhlo::ReduceWindowOp> {
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
    PatternRewriter& rewriter) const override {

    // avoid already fused
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    MhloFusionPattern pattern;

    // only support all pads or none pads
    size_t numPad = llvm::count_if(op.inputs(), [&](Value v) {
      return isa_and_nonnull<mhlo::PadOp>(v.getDefiningOp());
      });

    // handle all pad
    if (numPad == op.inputs().size()) {
      for (auto val : op.inputs()) {
        auto pad = cast<mhlo::PadOp>(op.inputs().front().getDefiningOp());
        // handle constant
        auto paddingValDefOp = pad.padding_value().getDefiningOp();
        if (IsSplatMhloConstant(paddingValDefOp)) {
          auto cloned = ReplicateDefiningOp(rewriter, pad, 1, 0);
          pattern.push_back(cloned);
        }

        pattern.push_back(pad);
      }
    } 

    // handle initial as a constant
    size_t idx = op.inputs().size();
    for (auto val : op.init_values()) {
      auto initialDefOp = val.getDefiningOp();
      if (IsSplatMhloConstant(initialDefOp)) {
        auto cloned = ReplicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
      }
      idx++;
    }

    pattern.push_back(op);

    auto fusion = creatMhloFusionFromPattern(rewriter, pattern);

    // add attr
    fusion->setAttr(getByteIRReduceFusionAttrName(), UnitAttr::get(fusion.getContext()));

    return success();
  }

};

struct ReduceFusionPass : public ReduceFusionBase<ReduceFusionPass> {

  ReduceFusionPass() : ReduceFusionBase() {}

  void runOnOperation() override {
    FuncOp funcOp = getOperation();

    OwningRewritePatternList patterns(funcOp.getContext());
    populateFuseReduceWindowPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("ReduceFusionPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateFuseReduceWindowPatterns(RewritePatternSet& patterns) {
  patterns.add<PadReduceWindowPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createReduceFusionPass() {
  return std::make_unique<ReduceFusionPass>();
}
