//===- DynamicShapeClustering.h -------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_DYNAMICSHAPECLUSTERING_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_DYNAMICSHAPECLUSTERING_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

constexpr StringRef getDynamicFuncAttrName() {
  return "__byteir_dynamic_sub_function";
}

std::unique_ptr<OperationPass<ModuleOp>>
createDynamicShapeClusteringPass(llvm::StringRef anchorTag = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_DYNAMICSHAPECLUSTERING_H