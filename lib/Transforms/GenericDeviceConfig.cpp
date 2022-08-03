//===- GenericDeviceConfig.cpp --------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/GenericDeviceConfig.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::byre;
using namespace llvm;

namespace {

static void AddGenericFuncAttrs(func::FuncOp func,
                                const std::string &computeName) {
  mlir::OpBuilder opBuilder(func);

  func->setAttr(getByrePrefix() + "kernel_name",
                opBuilder.getStringAttr(func.getName()));
  func->setAttr(getByreComputeName(), opBuilder.getStringAttr(computeName));
  func->setAttr(getByreForceComputeNameAttrName(), opBuilder.getUnitAttr());

  // trivial offsets
  SmallVector<int32_t> offsets;
  unsigned numArg = func.getNumArguments() + func.getNumResults();
  offsets.reserve(numArg);
  for (unsigned i = 0; i < numArg; ++i) {
    offsets.push_back(i);
  }

  func->setAttr(getByreArgOffsetAttrName(), opBuilder.getI32ArrayAttr(offsets));
}

// Main Pass
struct GenericDeviceConfigPass
    : public GenericDeviceConfigBase<GenericDeviceConfigPass> {

  GenericDeviceConfigPass(const std::string &anchor,
                          const std::string &computeName)
      : GenericDeviceConfigBase() {
    this->anchorAttr = anchor;
    this->computeName = computeName;
  }

  void runOnOperation() override {
    // early terminate if empty anchor or computeName
    if (anchorAttr.empty() || computeName.empty()) {
      return;
    }

    auto f = getOperation();

    // early terminate if func has no anchor
    if (!f->hasAttr(anchorAttr)) {
      return;
    }

    AddGenericFuncAttrs(f, computeName);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenericDeviceConfigPass(llvm::StringRef anchorTag,
                                    llvm::StringRef computeName) {
  return std::make_unique<GenericDeviceConfigPass>(anchorTag.str(),
                                                   computeName.str());
}
