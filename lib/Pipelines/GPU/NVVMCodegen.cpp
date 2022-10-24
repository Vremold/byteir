//===- NVVMCodegen.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/NVVMCodegen.h"
#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createNVVMCodegenPipeline(OpPassManager &pm) {
  // TODO add target for supporting different SMs
  // TODO use target to decide passes
  pm.addPass(createCollectGPUKernelPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createGPUToNVVMExtPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  addMultiCSEPipeline(pm, 3);
}
