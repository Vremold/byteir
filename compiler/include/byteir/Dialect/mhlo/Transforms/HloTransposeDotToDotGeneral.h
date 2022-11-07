//===- HloTransposeDotToDotGeneral.h --------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOTRANSPOSEDOTTODOTGENERAL_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOTRANSPOSEDOTTODOTGENERAL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

void populateHloTransposeDotToDotGeneralPattern(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>>
createHloTransposeDotToDotGeneralPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOTRANSPOSEDOTTODOTGENERAL_H