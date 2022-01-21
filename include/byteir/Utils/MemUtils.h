//===- MemUtils.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_MEMUTILS_H
#define BYTEIR_UTILS_MEMUTILS_H

#include "llvm/ADT/Optional.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"

namespace mlir {
class Attribute;
class MLIRContext;

Attribute wrapIntegerMemorySpace(unsigned space, MLIRContext* ctx);

llvm::Optional<Value> getDimSize(OpBuilder& b, Value val, unsigned idx);

} // namespace mlir

#endif // BYTEIR_UTILS_MEMUTILS_H
