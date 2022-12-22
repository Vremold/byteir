//===- CanonicalizeExt.cpp ----------------------------------------- C++ --===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/CanonicalizeExt.h"

#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Transforms/CondCanonicalize.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "./PassDetail.h"

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
    // put conditional canonicalizer too
    populateCondCanonicalizePatterns(owningPatterns);

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
    // Note using preOrder since we use reverse iterator later.
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { allNestedOps.push_back(op); });
    for (auto it = allNestedOps.rbegin(); it != allNestedOps.rend(); ++it) {
      Operation *op = *it;
      if (!op->use_empty())
        continue;
      if (wouldOpBeTriviallyDead(op)) {
        op->erase();
      } else {
        auto customOp = llvm::dyn_cast<mhlo::CustomCallOp>(op);
        if (customOp && !customOp.getHasSideEffect()) {
          op->erase();
        }
      }
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
