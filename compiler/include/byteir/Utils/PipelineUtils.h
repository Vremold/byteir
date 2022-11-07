//===- PipelineUtils.h -------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_PIPELINEUTILS_H
#define BYTEIR_UTILS_PIPELINEUTILS_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class ModuleOp;

void addCleanUpPassPipeline(OpPassManager &pm);

void addMultiCSEPipeline(OpPassManager &pm, unsigned cnt);

template <typename OpClass = ModuleOp, typename Builder, typename... Args>
void invokeOpPassPipelineBuilder(Builder builder, OpPassManager &pm,
                                 Args &&...args) {
  if (pm.getOpAnchorName() != OpPassManager::getAnyOpAnchorName() &&
      pm.getOpAnchorName() != OpClass::getOperationName()) {
    if (pm.getNesting() == OpPassManager::Nesting::Implicit) {
      builder(pm.nest<OpClass>(), std::forward<Args>(args)...);
      return;
    }
    llvm::report_fatal_error(
        llvm::Twine("Can't build pass pipeline on expected op type ") +
        OpClass::getOperationName() + " but got " + pm.getOpAnchorName());
  } else {
    builder(pm, std::forward<Args>(args)...);
  }
}
} // namespace mlir

#endif // BYTEIR_UTILS_PIPELINEUTILS_H
