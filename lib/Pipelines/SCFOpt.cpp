//===- SCFOpt.cpp ------------------------------------------------ C++---*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/SCFOpt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct SCFOptPipelinePass : public SCFOptPipelineBase<SCFOptPipelinePass> {
  SCFOptPipelinePass() : SCFOptPipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    // lower affine.apply in case there is some
    pm.addPass(createLowerAffinePass());
    pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
    pm.addNestedPass<func::FuncOp>(createCondCanonicalizePass());
    addCleanUpPassPipeline(pm);

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createSCFOptPipelinePass() {
  return std::make_unique<SCFOptPipelinePass>();
}
