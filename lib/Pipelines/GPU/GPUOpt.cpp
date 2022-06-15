//===- GPUOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "./PassDetail.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct GPUOptPipelinePass : public GPUOptPipelineBase<GPUOptPipelinePass> {
  GPUOptPipelinePass(const std::string &target) : GPUOptPipelineBase() {
    // TODO use target to decide passes
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    // apply PromotoBufferStack to func's with
    // getByteIRElementwiseFusionAttrName
    {
      OpPassManager anchoredPM(FuncOp::getOperationName());

      anchoredPM.addPass(createPromoteBuffersToStackPass(
          /*isSmallAlloc =*/[](Value) { return true; }));

      pm.addNestedPass<FuncOp>(createAnchoredFuncPipelinePass(
          getByteIRElementwiseFusionAttrName(), anchoredPM));
    }

    // Note: a trivial loop will be removed by canonicalizer
    // so no canonicalizer before used
    pm.addNestedPass<FuncOp>(
        createInsertTrivialSCFLoopPass(getByteIRElementwiseFusionAttrName()));

    // attach ToGPUAttr
    pm.addPass(createFuncTagPass(getByteIRElementwiseFusionAttrName(),
                                 getToGPUAttrName()));

    // TODO add device here after general copy finished
    // std::string deviceCudaAttr = "device:String:cuda";
    // pm.addPass(createFuncTagPass(getByteIRElementwiseFusionAttrName(),
    //                              deviceCudaAttr));

    std::string iteratorAttr =
        getLoopToSIMTAttrName().str() + ":String:" + getLinearIdXName().str();

    pm.addNestedPass<FuncOp>(
        createLoopTagPass(getByteIRElementwiseFusionAttrName(), iteratorAttr));

    pm.addPass(createConvertFuncToGPUPass(/*bs=*/{128, 1, 1}));

    addCleanUpPassPipeline(pm);
    pm.addNestedPass<FuncOp>(createGenPTXConfigPass());

    // soft-deprecated the following, since LoopFusionPass is buggy
    /*
    pm.addNestedPass<FuncOp>(createRewriteAffineToMemrefPass());
    pm.addNestedPass<FuncOp>(createCoalescedForToGPULaunchPass(128));
    addCleanUpPassPipeline(pm);
    pm.addPass(createLowerAffinePass());
    pm.addPass(createGpuLauchSinkIndexComputationsPass());
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(createGenPTXConfigPass());
    */

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
