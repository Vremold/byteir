//===- ByreHost.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ByreHost.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include <string>

using namespace mlir;
using namespace mlir::byre;

namespace {

  struct ByreHostPipelinePass : public ByreHostPipelineBase<ByreHostPipelinePass> {
  ByreHostPipelinePass(const std::string &entry, const std::string &device)
        : ByreHostPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->deviceFile = device;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createCollectFuncPass(
        byre::ByreDialect::getEntryPointFunctionAttrName()));

    std::string stringAttr = "device_file_name:String:" + deviceFile;
    pm.addPass(createFuncTagPass(stringAttr, entryFunc));

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};
} // namespace


std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByreHostPipelinePass(const std::string& entry, const std::string& deviceFile) {
  return std::make_unique<ByreHostPipelinePass>(entry, deviceFile);
}
