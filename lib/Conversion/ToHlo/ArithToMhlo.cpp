//===- ArithToMhlo.cpp ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
#include "byteir/Conversion/ToHlo/ArithToMhlo.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;

namespace {

#include "ArithToMhloPattern.inc"

struct ConvertArithToMhloPass
    : public ConvertArithToMhloBase<ConvertArithToMhloPass> {
  void runOnOperation() override {
    FuncOp op = getOperation();
    MLIRContext *context = op.getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createConvertArithToMhloPass() {
  return std::make_unique<ConvertArithToMhloPass>();
}
