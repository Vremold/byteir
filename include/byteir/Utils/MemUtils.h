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


// return rank 
llvm::Optional<int64_t> getRank(Value val);

llvm::Optional<Value> getDimSize(OpBuilder& b, Value val, unsigned idx);

// Create an alloc based on an existing Value 'val', with a given space.
// Return None, if not applicable.
llvm::Optional<Value> createAlloc(OpBuilder& b, Value val, unsigned space = 0);

} // namespace mlir

#endif // BYTEIR_UTILS_MEMUTILS_H
