//===- AttrUtils.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_ATTRUTILS_H
#define BYTEIR_UTILS_ATTRUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
class ElementsAttr;
class ShapedType;

/// Return a new ElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newShape'. 
llvm::Optional<ElementsAttr> reshapeSplatElementsAttr(
  ElementsAttr attr, llvm::ArrayRef<int64_t> newShape);

llvm::Optional<ElementsAttr> reshapeSplatElementsAttr(
  ElementsAttr attr, ShapedType newShape);

} // namespace mlir

#endif // BYTEIR_UTILS_ATTRUTILS_H
