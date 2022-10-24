//===- GPUOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
void createGPUOptPipelineImpl(OpPassManager &pm, const std::string &target) {
  // apply PromotoBufferStack to func's with
  // getByteIRElementwiseFusionAttrName
  {
    OpPassManager anchoredPM(func::FuncOp::getOperationName());

    anchoredPM.addPass(createPromoteBuffersToStackPass(
        /*isSmallAlloc =*/[](Value) { return true; }));

    pm.addNestedPass<func::FuncOp>(createAnchoredFuncPipelinePass(
        getByteIRElementwiseFusionAttrName(), anchoredPM));
  }

  // Note: a trivial loop will be removed by canonicalizer
  // so no canonicalizer before used
  pm.addNestedPass<func::FuncOp>(
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

  pm.addNestedPass<func::FuncOp>(
      createLoopTagPass(getByteIRElementwiseFusionAttrName(), iteratorAttr));

  pm.addPass(createConvertFuncToGPUPass(/*bs=*/{128, 1, 1}));

  addCleanUpPassPipeline(pm);
  pm.addNestedPass<func::FuncOp>(createGenPTXConfigPass());

  // soft-deprecated the following, since LoopFusionPass is buggy
  /*
  pm.addNestedPass<func::FuncOp>(createRewriteAffineToMemrefPass());
  pm.addNestedPass<func::FuncOp>(createCoalescedForToGPULaunchPass(128));
  addCleanUpPassPipeline(pm);
  pm.addPass(createLowerAffinePass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createGenPTXConfigPass());
  */
}

} // namespace

void mlir::createGPUOptPipeline(OpPassManager &pm,
                                const GPUOptPipelineOptions &options) {
  createGPUOptPipelineImpl(pm, options.target);
}
