//===- AttrUtils.cpp ------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/AttrUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace llvm;
using namespace mlir;

void mlir::parseConcatAttr(const std::string &concatAttr, std::string &attrName,
                           std::string &attrType, std::string &attrValue) {

  size_t first_semi = concatAttr.find(':');

  if (first_semi == std::string::npos) {
    attrName = concatAttr;
    attrType = "Unit";
  } else {
    attrName = concatAttr.substr(0, first_semi);
    size_t second_semi = concatAttr.find(':', first_semi + 1);
    attrType = concatAttr.substr(first_semi + 1, second_semi - first_semi - 1);
    if (second_semi != std::string::npos) {
      attrValue = concatAttr.substr(second_semi + 1);
    }
  }
}

void mlir::setParsedConcatAttr(Operation *op, const std::string &attrName,
                               const std::string &attrType,
                               const std::string &attrValue) {
  if (op == nullptr)
    return;

  auto ctx = op->getContext();
  if (attrType == "Unit") {
    op->setAttr(attrName, UnitAttr::get(ctx));
  } else if (attrType == "String") {
    op->setAttr(attrName, StringAttr::get(ctx, attrValue));
  } else if (attrType == "I32") {
    int intVal = std::stoi(attrValue);
    op->setAttr(attrName, IntegerAttr::get(IntegerType::get(ctx, 32), intVal));
  } else if (attrType == "F32") {
    float f32Val = std::stof(attrValue);
    op->setAttr(attrName, FloatAttr::get(Float32Type::get(ctx), f32Val));
  } else {
    op->emitOpError() << "unsupport attachAttr";
  }
}

llvm::Optional<ElementsAttr>
mlir::reshapeSplatElementsAttr(ElementsAttr attr,
                               llvm::ArrayRef<int64_t> newShape) {
  auto type = RankedTensorType::get(newShape, attr.getElementType());
  return reshapeSplatElementsAttr(attr, type);
}

llvm::Optional<ElementsAttr>
mlir::reshapeSplatElementsAttr(ElementsAttr attr, ShapedType newShape) {
  if (auto splat = attr.dyn_cast_or_null<SplatElementsAttr>()) {
    ElementsAttr ret = splat.reshape(newShape);
    return ret;
  }
  return None;
}

llvm::Optional<ElementsAttr> mlir::cloneSplatElementsAttr(ElementsAttr attr,
                                                          ShapedType type) {
  if (!attr.isSplat())
    return None;

  if (attr.isa<DenseFPElementsAttr>()) {
    ElementsAttr ret =
        DenseElementsAttr::get(type, attr.getSplatValue<FloatAttr>());
    return ret;
  } else if (attr.isa<DenseIntElementsAttr>()) {
    ElementsAttr ret =
        DenseElementsAttr::get(type, attr.getSplatValue<IntegerAttr>());
    return ret;
  }
  return None;
}