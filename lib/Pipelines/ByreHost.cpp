//===- ByreHost.cpp -------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::byre;

namespace {
void createByreHostPipelineImpl(OpPassManager &pm, const std::string &entryFunc,
                                const std::string &deviceFile,
                                const std::string &target) {
  pm.addPass(createCollectFuncPass(
      byre::ByreDialect::getEntryPointFunctionAttrName()));

  std::string stringAttr = "device_file_name:String:" + deviceFile;
  pm.addPass(createFuncTagPass(/*anchorTag=*/"", stringAttr, entryFunc));

  // currently use SetOpSpace + SetArgSpace to specify space here
  // TODO: later move to GPUOpt after general copy finish
  if (!target.empty()) {
    pm.addNestedPass<func::FuncOp>(createSetOpSpacePass(entryFunc, target));
    pm.addPass(createSetArgSpacePass(entryFunc, target, true));
  }
}
} // namespace

void mlir::createByreHostPipeline(OpPassManager &pm,
                                  const ByreHostPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByreHostPipelineImpl, pm, options.entryFunc,
                              options.deviceFile, options.target);
}
