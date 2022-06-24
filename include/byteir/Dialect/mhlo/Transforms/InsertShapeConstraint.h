//===- InsertShapeConstraint.h ------------------------------------- C++
//---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_INSERTSHAPECONSTRAINT_H
#define BYTEIR_TRANSFORMS_INSERTSHAPECONSTRAINT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createInsertShapeConstraintPass();

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_INSERTSHAPECONSTRAINT_H
