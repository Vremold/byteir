//===- NVVMCodegen.cpp -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/NVVMCodegen.h"
#include "./PassDetail.h"
#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

namespace {

  struct NVVMCodegenPipelinePass : public NVVMCodegenPipelineBase<NVVMCodegenPipelinePass> {
  NVVMCodegenPipelinePass()
    : NVVMCodegenPipelineBase() {
    // TODO use target to decide passes
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createCollectGPUKernelPass());
    pm.addPass(createLowerToCFGPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createGPUToNVVMExtPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    addMultiCSEPipeline(pm, 3);

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }

};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createNVVMCodegenPipelinePass() {
  return std::make_unique<NVVMCodegenPipelinePass>();
}