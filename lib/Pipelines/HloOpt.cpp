//===- HloOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/HloOpt.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Transforms/Passes.h"

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
    pm.addNestedPass<FuncOp>(createHloFolderPass());
    pm.addNestedPass<FuncOp>(createHloFolderPass());
    pm.addNestedPass<FuncOp>(createHloTransposeDotToDotGeneralPass());
    pm.addNestedPass<FuncOp>(createReduceFusionPass());

    // rewrite with constraint
    pm.addNestedPass<FuncOp>(createRewriteWithConstraintPass());

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
  pm.addNestedPass<FuncOp>(createClusterConstraintPass());
  pm.addPass(createFusionOutliningPass());

  // Fusion passes
  pm.addNestedPass<FuncOp>(createConvBackwardFusionPass());
  pm.addNestedPass<FuncOp>(createIOConvertFusionPass("mhlo.batch_norm_training",
                                                     "BatchNormTrainingOp"));
  pm.addNestedPass<FuncOp>(
      createIOConvertFusionPass("mhlo.batch_norm_grad", "BatchNormGradOp"));
  pm.addNestedPass<FuncOp>(createDotTransposeFusionPass());

  // expand tuple
  pm.addPass(CreateExpandHloTuplesPass(entry));
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(createFlattenTuplePass());

  // Element fusion (always last?)
  // Note: if outlineSingleElemwiseOp is set, element fusion must be the last
  // pass, since it will cluster every elemenwise op which is not fused yet into
  // the mhlo.fusion and outline it as an independent function later
  pm.addNestedPass<FuncOp>(createElementFusionPass(outlineSingleElemwiseOp));
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
