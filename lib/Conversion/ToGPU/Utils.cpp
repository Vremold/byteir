//===- Utils.cpp -------------------------------------------------- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToGPU/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::gpu;

GPUModuleOp mlir::getOrCreateGPUModule(ModuleOp m, StringRef moduleName) {
  for (auto &op : m.getBody()->without_terminator()) {
    if (auto gm = dyn_cast<gpu::GPUModuleOp>(op)) {
      if (gm.getName() == moduleName) {
        return gm;
      }
    }
  }

  // if not found, create one
  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  auto gm = builder.create<GPUModuleOp>(m.getLoc(), moduleName);
  return gm;
}

gpu::GPUFuncOp mlir::cloneFuncToGPUFunc(OpBuilder &builder, FuncOp func,
                                        gpu::GPUModuleOp gm) {

  builder.setInsertionPointToStart(gm.getBody());
  auto gpuFunc = builder.create<gpu::GPUFuncOp>(gm.getLoc(), func.getName(),
                                                func.getType());
  gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                   builder.getUnitAttr());

  Region &funcBody = func.body();
  Region &gpuFuncBody = gpuFunc.body();

  BlockAndValueMapping bvm;
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    bvm.map(func.getArgument(i), gpuFunc.getArgument(i));
  }

  Block &gpuEntryBlock = gpuFuncBody.front();
  Block &funcEntryBlock = funcBody.front();

  builder.setInsertionPointToStart(&gpuEntryBlock);
  for (auto &op : funcEntryBlock.without_terminator()) {
    builder.clone(op, bvm);
  }

  // create a terminator
  builder.create<gpu::ReturnOp>(gpuFunc.getLoc());
  return gpuFunc;
}
