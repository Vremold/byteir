//===- ShapeOpt.cpp --------------------------------------------*--- C++-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ShapeOpt.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createShapeOptPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createSetAssumingAlwaysTruePass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createInsertTieShapePass());
  pm.addNestedPass<func::FuncOp>(createInsertShapeConstraintPass());
  pm.addNestedPass<func::FuncOp>(createByteIRShapeReificationPass());
  addCleanUpPassPipeline(pm);
  pm.addNestedPass<func::FuncOp>(createResolveShapeConstraintPass());
  pm.addNestedPass<func::FuncOp>(createBoundedShapeInferencePass());
  pm.addPass(createCanonicalizerPass());
}
