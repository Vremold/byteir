//===- HloOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/HloOpt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common.h"
#include "mlir/Transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"

using namespace mlir;
using namespace mlir::mhlo;

namespace {

  struct HloOptPipelinePass : public HloOptPipelineBase<HloOptPipelinePass> {
  HloOptPipelinePass(const std::string& entry, const std::string& target)
    : HloOptPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createInlinerPass());
    pm.addPass(createCanonicalizerPass());

    addCleanUpPassPipeline(pm);

    // generic folding
    pm.addNestedPass<FuncOp>(createHloFolderPass());
    pm.addNestedPass<FuncOp>(createHloFolderPass());
    pm.addNestedPass<FuncOp>(createHloTransposeDotToDotGeneralPass());

    addCleanUpPassPipeline(pm);

    // add fusion patterns
    addGenericHloFusionPatterns(pm, entryFunc);

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }

};


} // namespace


void mlir::addGenericHloFusionPatterns(OpPassManager& pm, const std::string& entry) {

  // Dot Transpose fusion
  pm.addNestedPass<FuncOp>(createDotTransposeFusionPass());
  pm.addPass(createFusionOutliningPass());

  // expand tuple
  pm.addPass(CreateExpandHloTuplesPass(entry));
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(createFlattenTuplePass());

  // Element fusion (always last?)
  pm.addNestedPass<FuncOp>(createElementFusionPass());
  pm.addPass(createFusionOutliningPass());
  pm.addPass(createCSEPass());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createHloOptPipelinePass(const std::string& entry, const std::string& target) {
  return std::make_unique<HloOptPipelinePass>(entry, target);
}
