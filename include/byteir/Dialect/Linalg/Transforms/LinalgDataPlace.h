//===- LinalgDataPlace.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

constexpr StringRef getDataPlaceAttrName() { return "__byteir_data_place__"; }
constexpr int64_t getUnplacedSpace() { return -1; }

std::unique_ptr<OperationPass<FuncOp>> createLinalgDataPlacePass(
  ArrayRef<int64_t> spaces = {});

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H