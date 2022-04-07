//===- TotalBufferize.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/TotalBufferize.h"
#include "./PassDetail.h"
#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "byteir/Pipelines/Common.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct ByteIRTotalBufferizePipelinePass
    : public ByteIRTotalBufferizePipelineBase<
          ByteIRTotalBufferizePipelinePass> {
  ByteIRTotalBufferizePipelinePass() : ByteIRTotalBufferizePipelineBase() {}

  void runOnOperation() override {
    auto m = getOperation();
    OpPassManager pm(m.getOperationName());
    addByteIRTotalBufferizePatterns(pm);

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::addByteIRTotalBufferizePatterns(OpPassManager &pm) {
  pm.addPass(createConvertHloToLHloPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
  addCleanUpPassPipeline(pm);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByteIRTotalBufferizePipelinePass() {
  return std::make_unique<ByteIRTotalBufferizePipelinePass>();
}
