//===- BoundedShapeAnalysis.cpp -------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Analysis/BoundedShapeAnalysis.h"
#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bounded-shape-analysis"

using namespace mlir::shape_analysis;

namespace mlir {
LogicalResult BoundedShapeAnalysis::inferResultShapesWithKnowledges(
    Operation *op, ShapeKnowledges shapeKnowledges,
    ShapeValueKnowledges shapeValueKnowledges,
    llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
  InferBoundedReturnTypeComponents inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc = inferBoundedReturnTypeComponents(customCall.call_target_name());
  } else {
    inferFunc = inferBoundedReturnTypeComponents(op->getName().getStringRef());
  }

  if (nullptr == inferFunc) {
    // fallback to generic shape analysis
    return ShapeAnalysis::inferResultShapesWithKnowledges(
        op, shapeKnowledges, shapeValueKnowledges, results);
  }

  ValueTypeModificatoinRAII valueTypeModification;
  for (auto &&operand : op->getOperands()) {
    Type newType = operand.getType();
    if (auto shape = shapeKnowledges(operand)) {
      newType = shape;
    }
    if (newType != operand.getType()) {
      valueTypeModification.Push(operand, newType);
    }
  }

  ValueShapeRange range(op->getOperands(), shapeKnowledges,
                        shapeValueKnowledges);

  return inferFunc(op->getContext(), op->getLoc(), range,
                   op->getAttrDictionary(), op->getRegions(), results);
}

void BoundedShapeValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeValueLattice *> operands,
    ArrayRef<ShapeValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "bounded shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<mhlo::ComputeReshapeShapeOp>([&](Operation *op) {
        const ShapeValueLattice *shape = operands[1];
        assert(!shape->isUninitialized() && "operand must be initialized");
        ShapeValueLattice *lattice = results[0];
        Attribute attr = shape->getValue().getConstantValue();
        LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
        propagateIfChanged(lattice, lattice->join(mlir::dataflow::ConstantValue(
                                        attr, op->getDialect())));
      })
      .Default([&](Operation *op) {
        ShapeValueAnalysis::visitOperation(op, operands, results);
      });
}
} // namespace mlir
