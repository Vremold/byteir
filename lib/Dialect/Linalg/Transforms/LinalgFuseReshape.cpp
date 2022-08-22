//===- LinalgFuseReshape.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/LinalgFuseReshape.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

// FIXME: this pass is to replace the deprecated
// FoldReshapeOpsByLinearizationPass.
struct LinalgFuseReshapePass
    : public LinalgFuseReshapeBase<LinalgFuseReshapePass> {
  LinalgFuseReshapePass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    linalg::ControlFusionFn controlReshapeFusionFn =
        [](const OpResult &producer, OpOperand &consumer) {
          if (auto collapseOp =
                  producer.getDefiningOp<tensor::CollapseShapeOp>()) {
            if (!collapseOp.getSrc().getDefiningOp<linalg::LinalgOp>()) {
              return false;
            }
          }
          if (auto expandOp =
                  dyn_cast<tensor::ExpandShapeOp>(consumer.getOwner())) {
            if (expandOp->hasOneUse()) {
              OpOperand &use = *expandOp->getUses().begin();
              auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
              if (linalgOp && linalgOp.isOutputTensor(&use))
                return true;
            }
            return false;
          }
          return true;
        };
    RewritePatternSet fusionPatterns(context);
    linalg::populateFoldReshapeOpsByExpansionPatterns(fusionPatterns,
                                                      controlReshapeFusionFn);
    linalg::ControlFusionFn controlFn = [](const OpResult &producer,
                                           OpOperand &consumer) -> bool {
      if (isa<tensor::ExpandShapeOp>(producer.getDefiningOp())) {
        // Skip fusing the first operand.
        return consumer.getOperandNumber();
      }
      return true;
    };
    linalg::populateFoldReshapeOpsByCollapsingPatterns(
        fusionPatterns, [](const OpResult & /*producer*/,
                           OpOperand & /*consumer*/) { return true; });
    FrozenRewritePatternSet frozenFusionPatterns(std::move(fusionPatterns));
    (void)applyPatternsAndFoldGreedily(funcOp.getBody(), frozenFusionPatterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgFuseReshapePass() {
  return std::make_unique<LinalgFuseReshapePass>();
}
