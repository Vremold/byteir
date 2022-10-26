//===- ToLLVM.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/Host/ToLLVM.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Utils/PipelineUtils.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
// pass to collect llvm submodule which was never used outside `ToLLVMPipeline`
struct CollectLLVMSubmodulePass
    : public PassWrapper<CollectLLVMSubmodulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectLLVMSubmodulePass);

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
  }
};
} // namespace

void mlir::createToLLVMPipeline(OpPassManager &pm) {
  invokeOpPassPipelineBuilder(
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<CollectLLVMSubmodulePass>());
        // TODO: move bufferize passes to total bufferize and collect
        // memref.global operations into host module
        pm.addPass(arith::createConstantBufferizePass());
        pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
        pm.addNestedPass<func::FuncOp>(
            bufferization::createBufferDeallocationPass());

        pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(arith::createArithmeticExpandOpsPass());
        pm.addPass(createMemRefToLLVMPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());

        pm.addPass(createCanonicalizerPass());
      },
      pm);
}
