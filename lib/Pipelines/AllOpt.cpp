//===- AllOpt.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Pipelines/GPU/Passes.h"
#include "byteir/Pipelines/Passes.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByteirAllOptPipelinePass
    : public ByteirAllOptPipelineBase<ByteirAllOptPipelinePass> {
  ByteirAllOptPipelinePass(const std::string &entry, const std::string &target)
      : ByteirAllOptPipelineBase() {
    // TODO use target to decide passes
    this->entryFunc = entry;
    this->target = target;
  }

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    pm.addPass(createHloOptPipelinePass(entryFunc, target,
                                        true /*outlineSingleElemwiseOp*/));

    pm.addPass(createLinalgTensorOptPipelinePass(target));
    pm.addPass(createByteIRTotalBufferizePipelinePass());

    pm.addPass(createAffineOptPipelinePass());
    // optional, alternative to affine-opt
    // pm.addPass(createSCFOptPipelinePass());

    pm.addPass(createGPUOptPipelinePass(target));
    pm.addPass(createByreOptPipelinePass(entryFunc, true /*appendArgTypes*/,
                                         false /*disableMemoryPlanning*/));
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByteIRAllOptPipelinePass(const std::string &entry,
                                     const std::string &target) {
  return std::make_unique<ByteirAllOptPipelinePass>(entry, target);
}
