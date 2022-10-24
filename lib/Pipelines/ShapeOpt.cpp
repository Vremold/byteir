//===- ShapeOpt.cpp --------------------------------------------*--- C++-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ShapeOpt.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct ShapeOptPipelinePass
    : public ShapeOptPipelineBase<ShapeOptPipelinePass> {
  ShapeOptPipelinePass() : ShapeOptPipelineBase() {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    OpPassManager pm(funcOp.getOperationName());

    pm.addPass(createSetAssumingAlwaysTruePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createInsertTieShapePass());
    pm.addPass(createInsertShapeConstraintPass());
    pm.addPass(createByteIRShapeReificationPass());
    addCleanUpPassPipeline(pm);
    pm.addPass(createResolveShapeConstraintPass());
    pm.addPass(createBoundedShapeInferencePass());
    pm.addPass(createCanonicalizerPass());

    if (mlir::failed(runPipeline(pm, funcOp))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createShapeOptPipelinePass() {
  return std::make_unique<ShapeOptPipelinePass>();
}
