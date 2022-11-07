//===- bufferize.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Ace/Passes.h"
#include "byteir/Dialect/Ace/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Dialect/Lace/LaceDialect.h"

using namespace mlir;
using namespace bufferization;

namespace {
struct AceBufferizePass : public AceBufferizeBase<AceBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<ace::AceDialect>();
    options.bufferAlignment = 0; // TODO: set alignment

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    ace::AceDialect, lace::LaceDialect>();
    ace::registerBufferizableOpInterfaceExternalModels(registry);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAceBufferizePass() {
  return std::make_unique<AceBufferizePass>();
}
