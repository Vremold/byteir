//===- ShapeOpt.cpp --------------------------------------------*--- C++-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ShapeOpt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct ShapeOptPipelinePass
    : public ShapeOptPipelineBase<ShapeOptPipelinePass> {
  ShapeOptPipelinePass() : ShapeOptPipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addNestedPass<FuncOp>(createInsertTieShapePass());
    pm.addNestedPass<FuncOp>(createInsertShapeConstraintPass());
    pm.addNestedPass<FuncOp>(createShapeReificationPass());
    addCleanUpPassPipeline(pm);
    pm.addNestedPass<FuncOp>(createResolveShapeConstraintPass());
    pm.addNestedPass<FuncOp>(createBoundedShapeInferencePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createDynamicShapeClusteringPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createShapeOptPipelinePass() {
  return std::make_unique<ShapeOptPipelinePass>();
}
