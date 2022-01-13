//===- LinalgScopeTiling.h --------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGSCOPETILING_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGSCOPETILING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<FunctionPass> createLinalgScopeTilingPass(StringRef anchorTag = "",
  int64_t tileAxis = 0, int64_t tileSize = 0, 
  linalg::LinalgTilingLoopType loopType = linalg::LinalgTilingLoopType::Loops,
  StringRef distributionType = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGSCOPETILING_H