//===- AceDialect.cpp -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ace;

#include "byteir/Dialect/Ace/AceOpsDialect.cpp.inc"

template <typename T> static LogicalResult Verify(T op) { return success(); }

//===----------------------------------------------------------------------===//
// ace dialect.
//===----------------------------------------------------------------------===//

void AceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Ace/AceOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "byteir/Dialect/Ace/AceOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "byteir/Dialect/Ace/AceOpsAttributes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Ace/AceOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult mlir::ace::ConstOp::fold(ArrayRef<Attribute>) { return value(); }

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

Block *mlir::ace::OpaqueOp::addEntryBlock() {
  assert(body().empty() && "opaqueOp already has an entry block");
  auto *entry = new Block();
  body().push_back(entry);
  for (auto type : getOperandTypes()) {
    entry->addArgument(type, /*Location*/ getLoc());
  }
  return entry;
}