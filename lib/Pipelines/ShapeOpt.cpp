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
  invokeOpPassPipelineBuilder<func::FuncOp>(
      [](OpPassManager &pm) {
        pm.addPass(createSetAssumingAlwaysTruePass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createInsertTieShapePass());
        pm.addPass(createInsertShapeConstraintPass());
        pm.addPass(createByteIRShapeReificationPass());
        addCleanUpPassPipeline(pm);
        pm.addPass(createResolveShapeConstraintPass());
        pm.addPass(createBoundedShapeInferencePass());
        pm.addPass(createCanonicalizerPass());
      },
      pm);
}
