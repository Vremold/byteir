//===- GenericDeviceConfig.cpp --------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/GenericDeviceConfig.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/FuncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::byre;
using namespace llvm;

namespace {
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

    addGenericFuncAttrs(f, computeName);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenericDeviceConfigPass(llvm::StringRef anchorTag,
                                    llvm::StringRef computeName) {
  return std::make_unique<GenericDeviceConfigPass>(anchorTag.str(),
                                                   computeName.str());
}
