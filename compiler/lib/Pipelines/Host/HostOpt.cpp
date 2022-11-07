//===- HostOpt.cpp --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/Host/HostOpt.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
void createHostOptPipelineImpl(OpPassManager &pm, const std::string &fileName) {
  pm.addNestedPass<func::FuncOp>(createGenLLVMConfigPass(fileName));
  pm.addPass(createCollectFuncToLLVMPass());
}
} // namespace

void mlir::createHostOptPipeline(OpPassManager &pm,
                                 const HostOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createHostOptPipelineImpl, pm, options.fileName);
}
