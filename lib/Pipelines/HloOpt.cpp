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

using namespace mlir;
using namespace mlir::mhlo;

namespace {
void addGenericHloFusionPatterns(OpPassManager &pm, const std::string &entry,
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

void addCPUHloFusionPatterns(OpPassManager &pm, const std::string &entry) {
  // expand tuple
  pm.addPass(createExpandHloTuplesPass(entry));
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createFlattenTuplePass());

  // perform aggressive fusion
  pm.addNestedPass<func::FuncOp>(createHloAggressiveFusionPass());
  pm.addPass(createFusionOutliningPass());
  pm.addPass(createCSEPass());
}

void createHloOptPipelineImpl(OpPassManager &pm, const std::string &entryFunc,
                              const std::string &target,
                              bool outlineSingleElemwiseOp) {

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
  if (target == "CPU") {
    addCPUHloFusionPatterns(pm, entryFunc);
  } else {
    addGenericHloFusionPatterns(pm, entryFunc, outlineSingleElemwiseOp);
  }
}
} // namespace

void mlir::createHloOptPipeline(OpPassManager &pm,
                                const HloOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createHloOptPipelineImpl, pm, options.entryFunc,
                              options.target, options.outlineSingleElemwiseOp);
}
