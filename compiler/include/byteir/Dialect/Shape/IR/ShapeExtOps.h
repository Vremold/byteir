//===- ShapeExtOps.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_SHAPE_SHAPEEXTOPS_H
#define BYTEIR_DIALECT_SHAPE_SHAPEEXTOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h.inc"

#endif // BYTEIR_DIALECT_SHAPE_SHAPEEXTOPS_H
