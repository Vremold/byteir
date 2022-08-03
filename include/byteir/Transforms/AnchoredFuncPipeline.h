//===- AnchoredFuncPipeline.h ------------------------------------- C++ ---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_ANCHOREDFUNCPIPELINE_H
#define BYTEIR_TRANSFORMS_ANCHOREDFUNCPIPELINE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>>
createAnchoredFuncPipelinePass(llvm::StringRef anchorTag,
                               OpPassManager &otherPM);

std::unique_ptr<OperationPass<func::FuncOp>>
createAnchoredFuncPipelinePass(llvm::StringRef anchorTag = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_ANCHOREDFUNCPIPELINE_H
