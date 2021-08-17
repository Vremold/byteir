//===- GPUToNVVM.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_GPUTONVVM_H
#define BYTEIR_CONVERSION_GPUTONVVM_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

namespace gpu {
class GPUModuleOp;
} // namespace gpu


std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGPUToNVVMExtPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout);


} // namespace mlir

#endif // BYTEIR_CONVERSION_GPUTONVVM_H