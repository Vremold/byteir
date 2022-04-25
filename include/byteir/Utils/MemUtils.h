//===- MemUtils.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_UTILS_MEMUTILS_H
#define BYTEIR_UTILS_MEMUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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

// Returns the total amount of bits occupied by a value of MemRefType. This
// takes into account of memory layout constraints. Returns None if the size
// cannot be computed statically, e.g. if the type has a dynamic shape or if its
// elemental type does not have a known bit width.
llvm::Optional<int64_t> getSizeInBits(MemRefType t);

// Returns whether a value of MemRefType is static. It requires the shape,
// stride and offset are all static value.
bool isStatic(MemRefType t);

// Returns a new MemRefType with a new MemSpace 'attr'
MemRefType cloneMemRefTypeWithMemSpace(MemRefType t, Attribute attr);

// Reutrns a new MemRefType and remove MemSpace
MemRefType cloneMemRefTypeAndRemoveMemSpace(MemRefType t);

} // namespace mlir

#endif // BYTEIR_UTILS_MEMUTILS_H
