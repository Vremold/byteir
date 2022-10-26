//===- AllOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/AllOpt.h"
#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Pipelines/ShapeOpt.h"
#include "byteir/Pipelines/TotalBufferize.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {
void createByteIRAllOptPipelineImpl(OpPassManager &pm,
                                    const std::string &entryFunc,
                                    const std::string &target) {
  HloOptPipelineOptions hloOptOptions;
  hloOptOptions.entryFunc = entryFunc;
  hloOptOptions.target = target;
  hloOptOptions.outlineSingleElemwiseOp = true;
  createHloOptPipeline(pm, hloOptOptions);

  LinalgTensorOptPipelineOptions linalgTensorOptOptions;
  linalgTensorOptOptions.target = target;
  createLinalgTensorOptPipeline(pm, linalgTensorOptOptions);

  createByteIRTotalBufferizePipeline(pm);

  createAffineOptPipeline(pm);
  // optional, alternative to affine-opt
  // createSCFOptPipeline(pm);

  GPUOptPipelineOptions gpuOptOptions;
  gpuOptOptions.target = target;
  createGPUOptPipeline(pm, gpuOptOptions);

  ByreOptPipelineOptions byreOptOptions;
  byreOptOptions.entryFunc = entryFunc;
  byreOptOptions.appendArgTypes = true;
  byreOptOptions.disableMemoryPlanning = false;
  createByreOptPipeline(pm, byreOptOptions);
}
} // namespace

void mlir::createByteIRAllOptPipeline(
    OpPassManager &pm, const ByteIRAllOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByteIRAllOptPipelineImpl, pm,
                              options.entryFunc, options.target);
}
