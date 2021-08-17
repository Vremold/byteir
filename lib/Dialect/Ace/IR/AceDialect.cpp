//===- AceDialect.cpp -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/Builders.h"

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
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Ace/AceOps.cpp.inc"

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

Block *mlir::ace::OpaqueOp::addEntryBlock() {
  assert(body().empty() && "opaqueOp already has an entry block");
  auto *entry = new Block();
  body().push_back(entry);
  entry->addArguments(getOperandTypes());
  return entry;
}