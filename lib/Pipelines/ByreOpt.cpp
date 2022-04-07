//===- ByreOpt.cpp ---------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ByreOpt.h"
#include "./PassDetail.h"
#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByreOptPipelinePass : public ByreOptPipelineBase<ByreOptPipelinePass> {
  ByreOptPipelinePass(const std::string &entry, bool appendTypes)
      : ByreOptPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->appendArgTypes = appendTypes;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createFuncTagPass(
        getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()),
        entryFunc));

    pm.addPass(createConvertToByrePass(appendArgTypes));
    pm.addNestedPass<FuncOp>(createByreFoldPass());
    pm.addPass(createCSEPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByreOptPipelinePass(const std::string &entry, bool appendTypes) {
  return std::make_unique<ByreOptPipelinePass>(entry, appendTypes);
}
