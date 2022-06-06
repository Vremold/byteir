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

using namespace mlir;

namespace byteir {

void registerNonZeroInferBoundedReturnTypes();

inline void registerAllMhloInferBoundedReturnTypes() {
  registerNonZeroInferBoundedReturnTypes();
}

} // namespace byteir

#endif // BYTEIR_DIALECT_MHLO_BOUNDEDSHAPES_REGISTER_H