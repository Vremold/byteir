//===- Register.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_SHAPECONSTRAINTS_REGISTER_H
#define BYTEIR_DIALECT_MHLO_SHAPECONSTRAINTS_REGISTER_H

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

void registerEinsumShapeConstraints();
void registerDynamicPartitionShapeConstraints();

inline void registerAllMhloShapeConstraints() {
  registerEinsumShapeConstraints();
  registerDynamicPartitionShapeConstraints();
}

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_SHAPECONSTRAINTS_REGISTER_H