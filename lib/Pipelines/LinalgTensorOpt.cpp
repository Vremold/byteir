//===- LinalgTensorOpt.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "./PassDetail.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/GenericFusion.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

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

    addGenericLinalgElementwisePasses(pm);

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::addGenericLinalgElementwisePasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(
      createHloFusionToLinalgPass(getByteIRElementwiseFusionAttrName()));
  pm.addNestedPass<FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createFoldReshapeOpsByLinearizationPass());
  pm.addPass(createCSEPass());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgTensorOptPipelinePass(const std::string &target) {
  return std::make_unique<LinalgTensorOptPipelinePass>(target);
}