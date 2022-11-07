//===- GenLLVMConfig.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/FuncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

#define LLVM_JIT_OP "LLVMJITOp"
#define LLVM_FILE_NAME_ATTR "llvm_file_name"

namespace {
static void AttachLLVMConfigToAttr(func::FuncOp func,
                                   const std::string &fileName) {
  addGenericFuncAttrs(func, getByteIRLLVMJITOpKernelName().str());

  mlir::OpBuilder opBuilder(func);
  func->setAttr(byre::getByrePrefix() + LLVM_FILE_NAME_ATTR,
                opBuilder.getStringAttr(fileName));
}

struct GenLLVMConfigPass : public GenLLVMConfigBase<GenLLVMConfigPass> {
  GenLLVMConfigPass(const std::string &fileName) : GenLLVMConfigBase() {
    this->fileName = fileName;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr(getByteIRHloAggressiveFusionAttrName()) ||
        func->hasAttr(getByteIRElementwiseFusionAttrName())) {
      AttachLLVMConfigToAttr(func, this->fileName);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenLLVMConfigPass(const std::string &fileName) {
  return std::make_unique<GenLLVMConfigPass>(fileName);
}
