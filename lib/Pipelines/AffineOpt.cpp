//===- AffineOpt.cpp -------------------------------------------*--- C++-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createAffineOptPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
  pm.addNestedPass<func::FuncOp>(createSimplifyAffineStructuresPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createCondCanonicalizePass());
  addCleanUpPassPipeline(pm);
}
