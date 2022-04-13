//===- AttrUtils.cpp
//----------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/AttrUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace llvm;
using namespace mlir;

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
