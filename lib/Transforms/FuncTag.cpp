//===- FuncTag.cpp ------------------------------------------------- C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/FuncTag.h"
#include "./PassDetail.h"
#include "byteir/Utils/AttrUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace {

struct FuncTagPass : public FuncTagBase<FuncTagPass> {
  FuncTagPass(const std::string &anchor, const std::string &attach,
              const std::string &name)
      : FuncTagBase<FuncTagPass>() {
    this->anchorAttr = anchor;
    this->attachAttr = attach;
    this->funcName = name;
  }

  void parseAttachAttr(const std::string &attr) {
    size_t first_semi = attr.find(':');

    if (first_semi == std::string::npos) {
      attrName = attr;
      attrType = "Unit";
    } else {
      attrName = attr.substr(0, first_semi);
      size_t second_semi = attr.find(':', first_semi + 1);
      attrType = attr.substr(first_semi + 1, second_semi - first_semi - 1);
      if (second_semi != std::string::npos) {
        attrValue = attr.substr(second_semi + 1);
      }
    }
  }

  void runOnOperation() override {
    // early termination if
    // 1) no attachAttr or
    // 2) no specified funcName or anchorAttr
    if (attachAttr.empty() || (funcName.empty() && anchorAttr.empty()))
      return;

    parseConcatAttr(attachAttr, attrName, attrType, attrValue);

    if (attrName.empty())
      return;

    auto m = getOperation();
    auto ctx = m.getContext();

    for (auto funcOp : m.getOps<FuncOp>()) {
      if (funcOp.getName() == funcName || funcOp->hasAttr(anchorAttr)) {

        if (attrType == "Unit") {
          funcOp->setAttr(attrName, UnitAttr::get(ctx));
        } else if (attrType == "String") {
          funcOp->setAttr(attrName, StringAttr::get(ctx, attrValue));
        } else if (attrType == "I32") {
          int intVal = std::stoi(attrValue);
          funcOp->setAttr(attrName,
                          IntegerAttr::get(IntegerType::get(ctx, 32), intVal));
        } else if (attrType == "F32") {
          float f32Val = std::stof(attrValue);
          funcOp->setAttr(attrName,
                          FloatAttr::get(Float32Type::get(ctx), f32Val));
        } else {
          m.emitOpError() << "unsupport attachAttr";
        }
      }
    }
  }

  std::string attrName;
  std::string attrType;
  std::string attrValue;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuncTagPass(llvm::StringRef anchorTag, llvm::StringRef attachTag,
                        const std::string &funcName) {
  return std::make_unique<FuncTagPass>(anchorTag.str(), attachTag.str(),
                                       funcName);
}
