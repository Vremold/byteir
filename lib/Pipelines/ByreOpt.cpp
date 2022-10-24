//===- ByreOpt.cpp --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Conversion/LmhloToLace/LmhloToLace.h"
#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::byre;

namespace {
bool isLmhloConstant(mlir::Value value) {
  return llvm::any_of(value.getUses(), [&](OpOperand &use) {
    return llvm::isa<lmhlo::ConstantOp>(use.getOwner());
  });
}

void createByreOptPipelineImpl(OpPassManager &pm, const std::string &entryFunc,
                               bool appendArgTypes,
                               bool disableMemoryPlanning) {
  pm.addPass(createFuncTagPass(
      /*anchorTag=*/"",
      getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()),
      entryFunc));

  pm.addPass(createConvertFuncAndCallToByrePass(appendArgTypes));

  // only applied on entry point function
  OpPassManager anchoredPM(func::FuncOp::getOperationName());
  anchoredPM.addPass(createLmhloToLacePass());
  anchoredPM.addPass(createCanonicalizerPass());
  if (!disableMemoryPlanning) {
    // underlying memory of constant op cannot be reused
    anchoredPM.addPass(createMemoryPlanningPass(
        [&](mlir::Value v) { return !isLmhloConstant(v); }));
    anchoredPM.addPass(createCanonicalizerPass());
  }
  anchoredPM.addPass(createConvertLmhloToByrePass(appendArgTypes));
  anchoredPM.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createAnchoredFuncPipelinePass(
      ByreDialect::getEntryPointFunctionAttrName(), anchoredPM));

  pm.addPass(createCSEPass());
}
} // namespace

void mlir::createByreOptPipeline(OpPassManager &pm,
                                 const ByreOptPipelineOptions &options) {
  createByreOptPipelineImpl(pm, options.entryFunc, options.appendArgTypes,
                            options.disableMemoryPlanning);
}