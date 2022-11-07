//===- AceDialect.cpp -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ace;

#include "byteir/Dialect/Ace/AceOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ace Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct AceInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Operations in ace dialect are always legal to inline
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

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
  addInterfaces<AceInlinerInterface>();
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

OpFoldResult mlir::ace::ConstOp::fold(ArrayRef<Attribute>) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

Block *mlir::ace::OpaqueOp::addEntryBlock() {
  assert(getBody().empty() && "opaqueOp already has an entry block");
  auto *entry = new Block();
  getBody().push_back(entry);
  for (auto type : getOperandTypes()) {
    entry->addArgument(type, /*Location*/ getLoc());
  }
  return entry;
}