//===- InitAllPipelines.h -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_INITALLPIPELINES_H
#define BYTEIR_PIPELINES_INITALLPIPELINES_H

#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Pipelines/AllOpt.h"
#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Pipelines/ShapeOpt.h"
#include "byteir/Pipelines/TotalBufferize.h"

#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"
#include "byteir/Pipelines/GPU/NVVMCodegen.h"

#include "byteir/Pipelines/Host/HostOpt.h"
#include "byteir/Pipelines/Host/ToLLVM.h"

namespace mlir {

inline void registerAllByteIRCommonPipelines() {
  registerAffineOptPipeline();
  registerByreHostPipeline();
  registerByreOptPipeline();
  registerHloOptPipeline();
  registerLinalgTensorOptPipeline();
  registerSCFOptPipeline();
  registerShapeOptPipeline();
  registerByteIRTotalBufferizePipeline();
  registerByteIRAllOptPipeline();
}

inline void registerAllByteIRGPUPipelines() {
  registerGPUOptPipeline();
  registerNVVMCodegenPipeline();
  registerLinalgMemrefGPUPipeline();
  registerMatmulEpilogueGPUPipeline();
}

inline void registerAllByteIRHostPipelines() {
  registerHostOptPipeline();
  registerToLLVMPipeline();
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_INITALLPIPELINES_H
