//===- ReduceFusion.cpp ----------------------------------------*--- C++
//-*-===//
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
#include "byteir/Utils/Utils.h"
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
                                PatternRewriter &rewriter) const override {

    // avoid already fused
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    // handle a common, special case of ReduceWindow for 1 input, 1 init_values,
    // and 1 result
    if (op.inputs().size() == 1 && op.init_values().size() == 1 &&
        op.getResults().size() == 1) {
      if (auto pad = dyn_cast_or_null<mhlo::PadOp>(
              op.inputs().front().getDefiningOp())) {
        if (pad.padding_value() == op.init_values().front() &&
            isZeroAttribute(pad.interior_padding()) &&
            (!op.padding().hasValue() ||
             isZeroAttribute(op.padding().getValue()))) {
          // create a padding
          const auto &edge_padding_low = pad.edge_padding_low();
          const auto &edge_padding_high = pad.edge_padding_high();
          SmallVector<int64_t> newPadding;
          for (auto it : llvm::zip(edge_padding_low, edge_padding_high)) {
            newPadding.push_back(std::get<0>(it).getZExtValue());
            newPadding.push_back(std::get<1>(it).getZExtValue());
          }

          auto newPaddingAttr = DenseIntElementsAttr::get(
              RankedTensorType::get({edge_padding_low.size(), 2},
                                    rewriter.getI64Type()),
              newPadding);

          auto newOp = cast<mhlo::ReduceWindowOp>(rewriter.clone(*op));
          newOp.setOperand(0, pad.operand());
          newOp.paddingAttr(newPaddingAttr);
          rewriter.replaceOp(op, newOp->getResult(0));
          return success();
        }
      } else {
        return failure();
      }
    }

    // only support cases of all pads or none pads
    size_t numPad = llvm::count_if(op.inputs(), [&](Value v) {
      return isa_and_nonnull<mhlo::PadOp>(v.getDefiningOp());
    });

    MhloFusionPattern pattern;
    // handle the case of all pads
    if (numPad == op.inputs().size()) {
      for (auto val : op.inputs()) {
        auto pad = cast<mhlo::PadOp>(op.inputs().front().getDefiningOp());
        // handle pad of constant
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
    FuncOp funcOp = getOperation();

    OwningRewritePatternList patterns(funcOp.getContext());
    populateFuseReduceWindowPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
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

std::unique_ptr<OperationPass<FuncOp>> mlir::createReduceFusionPass() {
  return std::make_unique<ReduceFusionPass>();
}
