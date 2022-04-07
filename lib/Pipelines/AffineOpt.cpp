//===- AffineOpt.cpp ---------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/AffineOpt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common.h"
#include "byteir/Transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct AffineOptPipelinePass
    : public AffineOptPipelineBase<AffineOptPipelinePass> {
  AffineOptPipelinePass() : AffineOptPipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
    pm.addNestedPass<FuncOp>(createLoopCoalescingPass());
    pm.addNestedPass<FuncOp>(createSimplifyAffineStructuresPass());
    pm.addNestedPass<FuncOp>(createAffineLoopFusionExPass());
    pm.addNestedPass<FuncOp>(createInsertTrivialAffineLoopPass(
        getByteIRElementwiseFusionAttrName()));
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(createCMAEPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createAffineOptPipelinePass() {
  return std::make_unique<AffineOptPipelinePass>();
}
