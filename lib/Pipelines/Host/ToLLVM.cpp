//===- ToLLVM.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/Host/ToLLVM.h"
#include "./PassDetail.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace {

struct ToLLVMPipelinePass : public ToLLVMPipelineBase<ToLLVMPipelinePass> {
  ToLLVMPipelinePass() : ToLLVMPipelineBase() {}

  void collectAndInlineLLVMSubmodule(ModuleOp top) {
    SmallVector<Operation *> toRemove;
    SmallVector<ModuleOp> llvmSubmodule;
    for (auto &&op : *top.getBody()) {
      if (auto m = llvm::dyn_cast_or_null<ModuleOp>(&op)) {
        if (m->hasAttr(getByteIRLLVMModuleAttrName())) {
          llvmSubmodule.push_back(m);
        }
      }
      toRemove.push_back(&op);
    }
    for (auto &&sub : llvmSubmodule) {
      top.getBody()->getOperations().splice(top.getBody()->end(),
                                            sub.getBody()->getOperations());
    }
    for (auto &&op : toRemove) {
      op->erase();
    }
  }

  void runOnOperation() override {
    auto m = getOperation();

    if (!m->hasAttr(getByteIRLLVMModuleAttrName())) {
      collectAndInlineLLVMSubmodule(m);
    }

    OpPassManager pm(m.getOperationName());

    // TODO: move bufferize passes to total bufferize and collect
    // memref.global operations into host module
    pm.addPass(arith::createConstantBufferizePass());
    pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());

    pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createMemRefToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    pm.addPass(createCanonicalizerPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createToLLVMPipelinePass() {
  return std::make_unique<ToLLVMPipelinePass>();
}
