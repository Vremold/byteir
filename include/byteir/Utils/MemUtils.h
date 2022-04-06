//===- MemUtils.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_MEMUTILS_H
#define BYTEIR_UTILS_MEMUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
class Attribute;
class MLIRContext;

Attribute wrapIntegerMemorySpace(unsigned space, MLIRContext *ctx);

// return rank
llvm::Optional<int64_t> getRank(Value val);

llvm::Optional<Value> getDimSize(OpBuilder &b, Value val, unsigned idx);

// Create an alloc based on an existing Value 'val', with a given space.
// Return None, if not applicable.
llvm::Optional<Value> createAlloc(OpBuilder &b, Value val, unsigned space = 0);

// Get byte shift from the original allocation operation or function argument.
// Note that `shift` is different from `offset`, since `shift` is used for
// contiguous memory, while `offset` is used in multi-dimenstional situation.
// Return None, if val is not of type MemRefType or it could not be determined.
llvm::Optional<int64_t> getByteShiftFromAllocOrArgument(Value val);

} // namespace mlir

#endif // BYTEIR_UTILS_MEMUTILS_H
