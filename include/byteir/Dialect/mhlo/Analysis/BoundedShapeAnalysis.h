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

class BoundedShapeValueAnalysis : public ShapeValueAnalysis {
public:
  using ShapeValueAnalysis::ShapeValueAnalysis;

  // in consistent with ShapeValueAnalysis, add customized handle logic for
  // ops in mhlo dialect
  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override;
};

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_BOUNDEDSHAPEANALYSIS_H
