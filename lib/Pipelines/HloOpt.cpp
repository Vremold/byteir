//===- HloOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct HloOptPipelinePass : public HloOptPipelineBase<HloOptPipelinePass> {
  HloOptPipelinePass(const std::string &entry, const std::string &target,
                     bool outlineSingleElemwiseOp)
      : HloOptPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->target = target;
    this->outlineSingleElemwiseOp = outlineSingleElemwiseOp;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createInlinerPass());
    pm.addPass(createCanonicalizerPass());

    addCleanUpPassPipeline(pm);

    // generic folding
    pm.addNestedPass<func::FuncOp>(createHloFolderPass());
    pm.addNestedPass<func::FuncOp>(createHloFolderPass());
    pm.addNestedPass<func::FuncOp>(createHloTransposeDotToDotGeneralPass());
    pm.addNestedPass<func::FuncOp>(createReduceFusionPass());

    // rewrite with constraint
    pm.addNestedPass<func::FuncOp>(createRewriteWithConstraintPass());

    addCleanUpPassPipeline(pm);

    // add fusion patterns
    addGenericHloFusionPatterns(pm, entryFunc,
                                outlineSingleElemwiseOp.getValue());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::addGenericHloFusionPatterns(OpPassManager &pm,
                                       const std::string &entry,
                                       bool outlineSingleElemwiseOp) {

  // cluster constraint
  pm.addNestedPass<func::FuncOp>(createClusterConstraintPass());
  pm.addPass(createFusionOutliningPass());

  // Fusion passes
  pm.addNestedPass<func::FuncOp>(createConvBackwardFusionPass());
  pm.addNestedPass<func::FuncOp>(createIOConvertFusionPass());
  pm.addNestedPass<func::FuncOp>(createDotTransposeFusionPass());

  // expand tuple
  pm.addPass(createExpandHloTuplesPass(entry));
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createFlattenTuplePass());

  // Element fusion (always last?)
  // Note: if outlineSingleElemwiseOp is set, element fusion must be the last
  // pass, since it will cluster every elemenwise op which is not fused yet into
  // the mhlo.fusion and outline it as an independent function later
  pm.addNestedPass<func::FuncOp>(
      createElementFusionPass(outlineSingleElemwiseOp));
  pm.addPass(createFusionOutliningPass());
  pm.addPass(createCSEPass());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createHloOptPipelinePass(const std::string &entry,
                               const std::string &target,
                               bool outliningSingleElemwiseOp) {
  return std::make_unique<HloOptPipelinePass>(entry, target,
                                              outliningSingleElemwiseOp);
}
