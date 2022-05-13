//===- LinalgMemrefGPU.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"
#include "./PassDetail.h"
#include "byteir/Conversion/HloToLinalg/HloToLinalg.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/GenericFusion.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct LinalgMemrefGPUPipelinePass
    : public LinalgMemrefGPUPipelineBase<LinalgMemrefGPUPipelinePass> {
  LinalgMemrefGPUPipelinePass(const std::string & /*target*/)
      : LinalgMemrefGPUPipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

template <typename OTy>
void collectOp(FuncOp funcOp, SmallVectorImpl<Operation *> &collector) {
  for (auto op : funcOp.getOps<OTy>()) {
    collector.push_back(op);
  }
}

struct MatmulEpilogueGPUPipelinePass
    : public MatmulEpilogueGPUPipelineBase<MatmulEpilogueGPUPipelinePass> {
  MatmulEpilogueGPUPipelinePass(const std::string & /*target*/)
      : MatmulEpilogueGPUPipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();

    // TODO: add 3d tiling later
    // tile m-axis
    {
      SmallVector<Operation *> collection;
      for (auto funcOp : m.getOps<FuncOp>()) {
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

      // pass manager
      {
        OpPassManager pm(m.getOperationName());

        pm.addNestedPass<FuncOp>(createLinalgScopeTilingPass(0, 2));
        addCleanUpPassPipeline(pm);
        if (mlir::failed(runPipeline(pm, m))) {
          signalPassFailure();
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createMatmulEpilogueGPUPipelinePass(const std::string &target) {
  return std::make_unique<MatmulEpilogueGPUPipelinePass>(target);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgMemrefGPUPipelinePass(const std::string &target) {
  return std::make_unique<LinalgMemrefGPUPipelinePass>(target);
}
