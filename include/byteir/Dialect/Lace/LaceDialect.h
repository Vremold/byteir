//===- LaceDialect.h - MLIR Dialect for Mhlo Extension --------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LACE_LACEDIALECT_H
#define MLIR_DIALECT_LACE_LACEDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "byteir/Dialect/Lace/LaceOpInterfaces.h.inc"
#include "byteir/Dialect/Lace/LaceOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Lace/LaceOps.h.inc"

#endif // MLIR_DIALECT_LACE_LACEDIALECT_H
