//===- TotalBufferize.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_TOTALBUFFERIZE_H
#define BYTEIR_PIPELINES_TOTALBUFFERIZE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;

void addByteIRTotalBufferizePatterns(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>>
createByteIRTotalBufferizePipelinePass();

} // namespace mlir

#endif // BYTEIR_PIPELINES_TOTALBUFFERIZE_H