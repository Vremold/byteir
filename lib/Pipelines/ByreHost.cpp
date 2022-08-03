//===- ByreHost.cpp ---------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ByreHost.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <string>

using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByreHostPipelinePass
    : public ByreHostPipelineBase<ByreHostPipelinePass> {
  ByreHostPipelinePass(const std::string &entry, const std::string &deviceFile,
                       const std::string &target)
      : ByreHostPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->deviceFile = deviceFile;
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

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

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByreHostPipelinePass(const std::string &entry,
                                 const std::string &deviceFile,
                                 const std::string &target) {
  return std::make_unique<ByreHostPipelinePass>(entry, deviceFile, target);
}
