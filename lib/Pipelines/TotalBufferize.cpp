//===- TotalBufferize.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/TotalBufferize.h"
#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Ace/Passes.h"
#include "byteir/Dialect/Ace/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct ByteIRTotalBufferizePipelinePass
    : public ByteIRTotalBufferizePipelineBase<
          ByteIRTotalBufferizePipelinePass> {
  ByteIRTotalBufferizePipelinePass() : ByteIRTotalBufferizePipelineBase() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    ByteIRTotalBufferizePipelineBase::getDependentDialects(registry);
    registry.insert<ace::AceDialect, lace::LaceDialect>();
    ace::registerBufferizableOpInterfaceExternalModels(registry);
  }

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
  pm.addNestedPass<func::FuncOp>(createAceBufferizePass());
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  addCleanUpPassPipeline(pm);
  // clean-up possible redudant copy-removal from bufferization
  // TODO: enable it after fixing crash
  // pm.addNestedPass<func::FuncOp>(createCopyRemovalPass());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByteIRTotalBufferizePipelinePass() {
  return std::make_unique<ByteIRTotalBufferizePipelinePass>();
}
