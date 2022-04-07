//===- GPUOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPUOpt.h"
#include "./PassDetail.h"
#include "byteir/Conversion/AffineToGPU/AffineToGPU.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Pipelines/Common.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct GPUOptPipelinePass : public GPUOptPipelineBase<GPUOptPipelinePass> {
  GPUOptPipelinePass(const std::string &target) : GPUOptPipelineBase() {
    // TODO use target to decide passes
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addNestedPass<FuncOp>(createRewriteAffineToMemrefPass());
    pm.addNestedPass<FuncOp>(createCoalescedForToGPULaunchPass(128));
    addCleanUpPassPipeline(pm);
    pm.addPass(createLowerAffinePass());
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(createGenPTXConfigPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGPUOptPipelinePass(const std::string &target) {
  return std::make_unique<GPUOptPipelinePass>(target);
}
