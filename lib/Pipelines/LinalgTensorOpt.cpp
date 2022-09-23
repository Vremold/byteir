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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct LinalgTensorOptPipelinePass
    : public LinalgTensorOptPipelineBase<LinalgTensorOptPipelinePass> {
  LinalgTensorOptPipelinePass(const std::string &target)
      : LinalgTensorOptPipelineBase() {
    // TODO use target to decide passes
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    if (this->target.getValue() == "CPU") {
      addCPULinalgOptPasses(pm);
    } else {
      addGenericLinalgElementwisePasses(pm);
    }

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::addGenericLinalgElementwisePasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRElementwiseFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFuseReshapePass());
  pm.addPass(createCSEPass());
}

void mlir::addCPULinalgOptPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRHloAggressiveFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFuseReshapePass());
  pm.addPass(createCSEPass());
  // TODO: more opt passes
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgTensorOptPipelinePass(const std::string &target) {
  return std::make_unique<LinalgTensorOptPipelinePass>(target);
}
