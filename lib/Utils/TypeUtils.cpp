//===- TypeUtils.cpp ------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/TypeUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// append attribute to origin type's encoding
RankedTensorType mlir::appendTensorEncodingAttr(RankedTensorType origin,
                                                NamedAttribute attr) {
  if (!origin) {
    return origin;
  }
  llvm::SmallVector<NamedAttribute> originAttrs;
  if (auto dict = origin.getEncoding().dyn_cast_or_null<DictionaryAttr>()) {
    // copy origin type's encoding with DictionaryAttr
    originAttrs = llvm::to_vector(dict.getValue());
  }
  // note: if attr's name is same with origin attr, replace origin attr
  originAttrs.push_back(attr);
  DictionaryAttr dict =
      DictionaryAttr::get(attr.getValue().getContext(), originAttrs);
  // if origin type has an encoding which is not DictionaryAttr, replace it.
  return RankedTensorType::get(origin.getShape(), origin.getElementType(),
                               dict);
}
