//===- LinalgMemrefGPU.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

void createLinalgMemrefGPUPipelineImpl(OpPassManager & /* pm */,
                                       const std::string & /*target*/) {
  // TODO?
}

template <typename OTy>
void collectOp(func::FuncOp funcOp, SmallVectorImpl<Operation *> &collector) {
  for (auto op : funcOp.getOps<OTy>()) {
    collector.push_back(op);
  }
}

// preprocess pass which was never used outside `MatmulEpilogueGPUPipeline`
struct MatmulEpilogueGPUPipelinePreprocessPass
    : PassWrapper<MatmulEpilogueGPUPipelinePreprocessPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MatmulEpilogueGPUPipelinePreprocessPass)

  void runOnOperation() override {
    auto m = getOperation();

    // TODO: add 3d tiling later
    // tile m-axis
    {
      SmallVector<Operation *> collection;
      for (auto funcOp : m.getOps<func::FuncOp>()) {
        if (!funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) ||
            !funcOp.isPrivate()) {
          continue;
        }
        collectOp<linalg::MatmulOp>(funcOp, collection);
      }

      // early termination if no collection
      if (collection.empty())
        return;

      auto ctx = m.getContext();
      for (auto op : collection) {
        op->setAttr(getScopeTilingAnchorAttrName(), UnitAttr::get(ctx));
      }
    }
  }
};

void createMatmulEpilogueGPUPipelineImpl(OpPassManager &pm,
                                         const std::string &target) {
  pm.addPass(std::make_unique<MatmulEpilogueGPUPipelinePreprocessPass>());
  pm.addNestedPass<func::FuncOp>(createLinalgScopeTilingPass(0, 2));
  addCleanUpPassPipeline(pm);
}

} // namespace

void mlir::createLinalgMemrefGPUPipeline(
    OpPassManager &pm, const LinalgMemrefGPUPipelineOptions &options) {
  createLinalgMemrefGPUPipelineImpl(pm, options.target);
}

void mlir::createMatmulEpilogueGPUPipeline(
    OpPassManager &pm, const MatmulEpilogueGPUPipelineOptions &options) {
  createMatmulEpilogueGPUPipelineImpl(pm, options.target);
}