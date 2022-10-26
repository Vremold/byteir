//===- SCFOpt.cpp ------------------------------------------------ C++---*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createSCFOptPipeline(OpPassManager &pm) {
  invokeOpPassPipelineBuilder(
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
        // lower affine.apply in case there is some
        pm.addPass(createLowerAffinePass());
        pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
        pm.addNestedPass<func::FuncOp>(createCondCanonicalizePass());
        addCleanUpPassPipeline(pm);
      },
      pm);
}
