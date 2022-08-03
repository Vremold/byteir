//===- ByreDialect.h - MLIR Dialect for ByteIR Runtime ----------*- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines the Runtime-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BYRE_BYREDIALECT_H
#define MLIR_DIALECT_BYRE_BYREDIALECT_H

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func

namespace byre {

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

// Adds a `byre.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

} // end namespace byre
} // end namespace mlir

#include "byteir/Dialect/Byre/ByreOpsDialect.h.inc"

#include "byteir/Dialect/Byre/ByreOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Byre/ByreOps.h.inc"

#include "byteir/Dialect/Byre/ByreEnums.h.inc"

#endif // MLIR_DIALECT_BYRE_BYREDIALECT_H
