//===- AttrUtils.h ------------------------------------------------ C++---===//
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
class Operation;
class ShapedType;

// parse concatAttr into attrName:attrType:attrValue
void parseConcatAttr(const std::string &concatAttr, std::string &attrName,
                     std::string &attrType, std::string &attrValue);

void setParsedConcatAttr(Operation *op, const std::string &attrName,
                         const std::string &attrType,
                         const std::string &attrValue);

/// Return a new ElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newShape'.
llvm::Optional<ElementsAttr>
reshapeSplatElementsAttr(ElementsAttr attr, llvm::ArrayRef<int64_t> newShape);

llvm::Optional<ElementsAttr> reshapeSplatElementsAttr(ElementsAttr attr,
                                                      ShapedType newShape);

llvm::Optional<ElementsAttr> cloneSplatElementsAttr(ElementsAttr attr,
                                                    ShapedType newShape);

} // namespace mlir

#endif // BYTEIR_UTILS_ATTRUTILS_H