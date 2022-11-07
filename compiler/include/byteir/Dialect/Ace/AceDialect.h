//===- AceDialect.h - MLIR Dialect for Mhlo Extension ---------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ACE_ACEDIALECT_H
#define MLIR_DIALECT_ACE_ACEDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsAttributes.h.inc"

#include "byteir/Dialect/Ace/AceOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Ace/AceOps.h.inc"

#endif // MLIR_DIALECT_ACE_ACEDIALECT_H
