//===- ElementFusion.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_ELEMENTFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_ELEMENTFUSION_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {

constexpr StringRef getByteIRElementwiseFusionAttrName() {
  return "__byteir_elementwise_fusion__";
}

std::unique_ptr<OperationPass<FuncOp>>
createElementFusionPass(bool clusterSingleElemwiseOp = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_ELEMENTFUSION_H