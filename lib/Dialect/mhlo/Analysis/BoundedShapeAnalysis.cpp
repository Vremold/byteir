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
#include "llvm/Support/Debug.h"

using namespace mlir::shape_analysis;

namespace mlir {
LogicalResult BoundedShapeAnalysis::inferResultShapesWithKnowledges(
    Operation *op, ShapeKnowledges shapeKnowledges,
    ShapeValueKnowledges shapeValueKnowledges,
    llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
  InferBoundedReturnTypes inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc = inferBoundedReturnTypes(customCall.call_target_name());
  } else {
    inferFunc = inferBoundedReturnTypes(op->getName().getStringRef());
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
    if (auto value = shapeValueKnowledges(operand)) {
      if (auto ty = newType.dyn_cast_or_null<RankedTensorType>()) {
        newType = RankedTensorType::get(
            ty.getShape(), ty.getElementType(),
            DictionaryAttr::get(
                op->getContext(),
                {NamedAttribute(StringAttr::get(op->getContext(),
                                                getBoundedShapeDenseAttrName()),
                                value)}));
      }
    }
    if (newType != operand.getType()) {
      valueTypeModification.Push(operand, newType);
    }
  }
  llvm::SmallVector<Type> inferredType;
  LogicalResult inferStatus =
      inferFunc(op->getContext(), op->getLoc(), op->getOperands(),
                op->getAttrDictionary(), op->getRegions(), inferredType);
  if (succeeded(inferStatus)) {
    results.assign(llvm::to_vector(
        llvm::map_range(inferredType, [](mlir::Type t) -> ShapedTypeComponents {
          if (auto st = t.dyn_cast_or_null<ShapedType>())
            return st;
          return {};
        })));
  }
  return inferStatus;
}
} // namespace mlir
