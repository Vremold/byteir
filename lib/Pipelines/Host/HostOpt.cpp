//===- HostOpt.cpp --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/Host/HostOpt.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "byteir/Conversion/ToLLVM/ToLLVM.h"

using namespace mlir;

namespace {

struct HostOptPipelinePass : public HostOptPipelineBase<HostOptPipelinePass> {
  HostOptPipelinePass(const std::string &fileName) : HostOptPipelineBase() {
    this->fileName = fileName;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());
    pm.addNestedPass<func::FuncOp>(createGenLLVMConfigPass(this->fileName));
    pm.addPass(createCollectFuncToLLVMPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createHostOptPipelinePass(const std::string &fileName) {
  return std::make_unique<HostOptPipelinePass>(fileName);
}
