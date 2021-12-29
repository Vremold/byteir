//===- Fold.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_TRANSFORMS_FOLD_H
#define BYTEIR_DIALECT_BYRE_TRANSFORMS_FOLD_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<FunctionPass>
createByreFoldPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_BYRE_TRANSFORMS_FOLD_H