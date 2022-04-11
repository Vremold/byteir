//===- Utils.h ---------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOGPU_UTILS_H
#define BYTEIR_CONVERSION_TOGPU_UTILS_H

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class OpBuilder;

enum class GPUIndexType : uint32_t {
  thread_id = 0,
  block_id = 1,
  linear_id = 2,
};

// get GPUModuleOp or create one if there is none with moduleName
gpu::GPUModuleOp getOrCreateGPUModule(ModuleOp m, llvm::StringRef moduleName);

// clone FuncOp with body into GPUFuncOp
gpu::GPUFuncOp cloneFuncToGPUFunc(OpBuilder &builder, FuncOp func,
                                  gpu::GPUModuleOp gm);

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOGPU_UTILS_H
