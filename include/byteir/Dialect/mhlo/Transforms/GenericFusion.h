//===- GenericFusion.h ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSION_H

#include "mlir/Pass/Pass.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {

constexpr StringRef getByteIRElementwiseFusionAttrName() {
  return "__byteir_elementwise_fusion__";
}

constexpr StringRef getByteIRMatmulEpilogueFusionAttrName() {
  return "__byteir_matmul_epilogue_fusion__";
}

std::unique_ptr<OperationPass<FuncOp>>
createElementFusionPass(bool clusterSingleElemwiseOp = false);

std::unique_ptr<OperationPass<FuncOp>> createMatmulEpilogueFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSION_H
