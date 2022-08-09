//===- BoundedShapeAnalysis.h ---------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_ANALYSIS_BOUNDEDSHAPEANALYSIS_H
#define BYTEIR_DIALECT_MHLO_ANALYSIS_BOUNDEDSHAPEANALYSIS_H

#include "byteir/Analysis/ShapeAnalysis.h"

namespace mlir {
class BoundedShapeAnalysis : public ShapeAnalysis {
public:
  using ShapeAnalysis::ShapeAnalysis;

  LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) override;
};

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_BOUNDEDSHAPEANALYSIS_H
