//===- LayoutTransformation.h ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_LAYOUTTRANSFORMATION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_LAYOUTTRANSFORMATION_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
class RewritePatternSet;
namespace func {
class FuncOp;
} // namespace func

void populateLayoutTransformationPattern(RewritePatternSet &patterns,
                                         std::string targetLayout);

std::unique_ptr<OperationPass<func::FuncOp>>
createLayoutTransformationPass(std::string target_layout = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_LAYOUTTRANSFORMATION_H