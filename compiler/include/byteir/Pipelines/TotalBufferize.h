//===- TotalBufferize.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_TOTALBUFFERIZE_H
#define BYTEIR_PIPELINES_TOTALBUFFERIZE_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

void createByteIRTotalBufferizePipeline(OpPassManager &pm);

inline void registerByteIRTotalBufferizePipeline() {
  PassPipelineRegistration<>(
      "byteir-total-bufferize",
      "Performs all bufferization, including mhlo to lmhlo",
      createByteIRTotalBufferizePipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_TOTALBUFFERIZE_H