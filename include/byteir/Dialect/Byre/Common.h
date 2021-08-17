//===- Common.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BYRE_COMMON_H
#define MLIR_DIALECT_BYRE_COMMON_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace byre {

// byre.compute attribute name
inline llvm::StringRef getByreComputeName() {
  return "byre_compute_name";
}

inline llvm::StringRef getByreElementwiseFusionName() {
  return "byre_elementwise_fusion";
}


} // end namespace byre
} // end namespace mlir

#endif // MLIR_DIALECT_BYRE_COMMON_H