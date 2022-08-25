//===- ToLLVM.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_TOLLVM_H
#define BYTEIR_CONVERSION_TOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class ModuleOp;

constexpr StringRef getByteIRLLVMJITOpKernelName() { return "LLVMJITOp"; }

constexpr StringRef getByteIRLLVMModuleAttrName() {
  return "byteir.llvm_module";
}

std::unique_ptr<OperationPass<func::FuncOp>>
createGenLLVMConfigPass(const std::string &fileName = "host_kernels.ll");

std::unique_ptr<OperationPass<ModuleOp>> createCollectFuncToLLVMPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOLLVM_H