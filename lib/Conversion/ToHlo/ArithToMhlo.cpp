//===- ArithToMhlo.cpp ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToHlo/ArithToMhlo.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;

namespace {

#include "ArithToMhloPattern.inc"

struct ConvertArithToMhloPass
    : public ConvertArithToMhloBase<ConvertArithToMhloPass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    MLIRContext *context = op.getContext();
    RewritePatternSet patterns(context);
    populateWithGenerated(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(op, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertArithToMhloPass() {
  return std::make_unique<ConvertArithToMhloPass>();
}
