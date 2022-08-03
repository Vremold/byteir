//===- SetArgShape.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_SETARGSHAPE_H
#define BYTEIR_TRANSFORMS_SETARGSHAPE_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>> createSetArgShapePass();
std::unique_ptr<OperationPass<ModuleOp>>
createSetArgShapePass(int dim, int size, std::string entryFuncName,
                      std::string argAttrName);
std::unique_ptr<OperationPass<ModuleOp>>
createSetArgShapePass(int dim, int size, std::string entryFuncName,
                      std::function<bool(BlockArgument)> shouldSetShape);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_SETARGSHAPE_H