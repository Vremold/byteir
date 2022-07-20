//===- CanonicalizeExt.cpp ----------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CanonicalizeExt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace {

struct CanonicalizeExtPass : public CanonicalizeExtBase<CanonicalizeExtPass> {
  CanonicalizeExtPass() = default;
  CanonicalizeExtPass(const GreedyRewriteConfig &config,
                      ArrayRef<std::string> disabledPatterns,
                      ArrayRef<std::string> enabledPatterns) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);
    mhlo::getCanonicalizationExtPatterns(owningPatterns, context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() override {
    Operation *operation = getOperation();

    // TODO: The ideal way of adding mhlo.custom_call dce logic is to
    // integrating it into applyPatternsAndFoldGreedily.
    // Side effect is only an attribute of CustomCallOp, not an interface. It
    // should be specially handled.
    std::vector<Operation *> allNestedOps;
    operation->walk([&](Operation *op) { allNestedOps.push_back(op); });
    for (auto it = allNestedOps.rbegin(); it != allNestedOps.rend(); ++it) {
      Operation *op = *it;
      if (!op->use_empty())
        continue;
      if (wouldOpBeTriviallyDead(op))
        op->erase();
      auto customOp = llvm::dyn_cast<mhlo::CustomCallOp>(op);
      if (customOp && !customOp.has_side_effect())
        op->erase();
    }

    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    (void)applyPatternsAndFoldGreedily(operation, patterns, config);
  }

  FrozenRewritePatternSet patterns;
};

} // namespace

std::unique_ptr<Pass> mlir::createCanonicalizeExtPass() {
  return std::make_unique<CanonicalizeExtPass>();
}

std::unique_ptr<Pass>
mlir::createCanonicalizeExtPass(const GreedyRewriteConfig &config,
                                ArrayRef<std::string> disabledPatterns,
                                ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<CanonicalizeExtPass>(config, disabledPatterns,
                                               enabledPatterns);
}
