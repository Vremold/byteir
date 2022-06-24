//===- ShapeExtOps.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_SHAPEXTOPS_H
#define MLIR_DIALECT_SHAPE_SHAPEXTOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "byteir/Dialect/Shape/ShapeExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Shape/ShapeExtOps.h.inc"

#endif // MLIR_DIALECT_SHAPE_SHAPEXTOPS_H
