//===- LinalgTensorOpt.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgFuseReshape.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void addGenericLinalgElementwisePasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRElementwiseFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFuseReshapePass());
  pm.addPass(createCSEPass());
}

void addCPULinalgOptPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRHloAggressiveFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFuseReshapePass());
  pm.addPass(createCSEPass());
  // TODO: more opt passes
}

void createLinalgTensorOptPipelineImpl(OpPassManager &pm,
                                       const std::string &target) {
  if (target == "CPU") {
    addCPULinalgOptPasses(pm);
  } else {
    addGenericLinalgElementwisePasses(pm);
  }
}
} // namespace

void mlir::createLinalgTensorOptPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgTensorOptPipelineImpl, pm,
                              options.target);
}
