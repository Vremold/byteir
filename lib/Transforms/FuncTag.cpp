//===- FuncTag.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/FuncTag.h"
#include "./PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace {

struct FuncTagPass : public FuncTagBase<FuncTagPass> {
  FuncTagPass(const std::string& tag, const std::string& name)
    : FuncTagBase<FuncTagPass>() {
    this->attachAttr = tag;
    this->funcName = name;
  }

  void runOnOperation() override {
    if (attachAttr.empty()) return;

    auto m = getOperation();
    auto ctx = m.getContext();

    for (auto funcOp : m.getOps<FuncOp>()) {
      if (funcName.empty() ||
        funcOp.getName() == funcName) {
        funcOp->setAttr(attachAttr, UnitAttr::get(ctx));
      }
    }
  }

};


} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuncTagPass(const std::string& attachTag, const std::string& funcName) {
  return std::make_unique<FuncTagPass>(attachTag, funcName);
}
