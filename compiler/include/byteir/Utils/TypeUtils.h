//===- TypeUtils.h ------------------------------------------------ C++---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_TYPEUTILS_H
#define BYTEIR_UTILS_TYPEUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

// append attribute to origin type's encoding
// note: if origin has non-DictionaryAttr, will replace it.
RankedTensorType appendTensorEncodingAttr(RankedTensorType origin,
                                          NamedAttribute attr);

} // namespace mlir

#endif // BYTEIR_UTILS_TYPEUTILS_H
