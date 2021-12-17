//===- ArithOptimize.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/transforms/ArithOptimize.h"
#include "PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace llvm;

namespace {
#include "transforms/ArithOptimizePattern.inc"

struct MhloArithOptPass : public MhloArithOptBase<MhloArithOptPass> {
  void runOnFunction() override final;
};

void MhloArithOptPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateMhloArithOptPatterns(patterns);
  LogicalResult status =
      applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  if (failed(status)) {
    signalPassFailure();
  }
}
} // namespace

void mlir::populateMhloArithOptPatterns(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

std::unique_ptr<FunctionPass> mlir::createMhloArithOptPass() {
  return std::make_unique<MhloArithOptPass>();
}
