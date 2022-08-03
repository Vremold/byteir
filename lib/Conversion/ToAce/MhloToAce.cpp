//===- MhloToAce.cpp ------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
#include "byteir/Conversion/ToAce/MhloToAce.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::ace;

namespace {

#include "MhloToAceActivationPattern.inc"

void populateFuseMhloToAceActivationPatterns(MLIRContext *context,
                                             RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
}

struct ConvertMhloToAcePass
    : public ConvertMhloToAceBase<ConvertMhloToAcePass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    MLIRContext *context = op.getContext();
    RewritePatternSet patterns(context);
    populateFuseMhloToAceActivationPatterns(context, &patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertMhloToAcePass() {
  return std::make_unique<ConvertMhloToAcePass>();
}
