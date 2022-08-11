//===- Register.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_BOUNDEDSHAPES_REGISTER_H
#define BYTEIR_DIALECT_MHLO_BOUNDEDSHAPES_REGISTER_H

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

void registerDynamicPartitionInferBoundedReturnTypeComponents();

void registerNonZeroInferBoundedReturnTypeComponents();

void registerWhereInferBoundedReturnTypeComponents();

void registerDynamicBroadcastInDimInferBoundedReturnTypeComponents();

void registerDynamicReshapeInferBoundedReturnTypeComponents();

void registerReduceInferBoundedReturnTypeComponents();

inline void registerAllMhloInferBoundedReturnTypeComponents() {
  registerDynamicPartitionInferBoundedReturnTypeComponents();
  registerNonZeroInferBoundedReturnTypeComponents();
  registerWhereInferBoundedReturnTypeComponents();
  registerDynamicBroadcastInDimInferBoundedReturnTypeComponents();
  registerDynamicReshapeInferBoundedReturnTypeComponents();
  registerReduceInferBoundedReturnTypeComponents();
}

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_BOUNDEDSHAPES_REGISTER_H
