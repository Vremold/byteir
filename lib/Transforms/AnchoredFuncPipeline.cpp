//===- AnchoredFuncPipeline.cpp ------------------------------------ C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/AnchoredFuncPipeline.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace {

struct AnchoredFuncPipelinePass
    : public AnchoredFuncPipelineBase<AnchoredFuncPipelinePass> {

  explicit AnchoredFuncPipelinePass(const std::string &anchor)
      : AnchoredFuncPipelineBase<AnchoredFuncPipelinePass>(),
        pm(func::FuncOp::getOperationName()) {
    this->anchorAttr = anchor;
  }

  AnchoredFuncPipelinePass(const std::string &anchor, OpPassManager &otherPM)
      : AnchoredFuncPipelineBase<AnchoredFuncPipelinePass>(), pm(otherPM) {
    this->anchorAttr = anchor;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    pm.getDependentDialects(registry);
  }

  void runOnOperation() override {
    if (anchorAttr.empty()) {
      return;
    }

    auto f = getOperation();

    if (!f->hasAttr(anchorAttr)) {
      return;
    }

    if (mlir::failed(runPipeline(pm, f))) {
      signalPassFailure();
    }
  }

  OpPassManager pm;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAnchoredFuncPipelinePass(llvm::StringRef anchorTag,
                                     OpPassManager &otherPM) {
  return std::make_unique<AnchoredFuncPipelinePass>(anchorTag.str(), otherPM);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAnchoredFuncPipelinePass(llvm::StringRef anchorTag) {
  return std::make_unique<AnchoredFuncPipelinePass>(anchorTag.str());
}
