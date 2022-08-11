//===- BoundedShapeAnalysis.cpp -------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Analysis/BoundedShapeAnalysis.h"
#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Utils/Utils.h"
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

  //  if return Attr{nullptr}, Type{nullptr} directly, ShapeAdaptor would try
  //  dync_cast<> which cause crash
  auto wrapperShapeKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto type = shapeKnowledges(v)) {
      return type;
    }
    return nullptr;
  };
  auto wrapperShapeValueKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto attr = shapeValueKnowledges(v)) {
      return attr;
    }
    return nullptr;
  };
  ValueShapeRange range(op->getOperands(), wrapperShapeKnowledges,
                        wrapperShapeValueKnowledges);

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
        // in some cases, the shape in computeReshapeShapeOp is dense<[-1, x,
        // ....]>, we need calculate firstly
        do {
          auto denseInt = attr.dyn_cast_or_null<DenseIntElementsAttr>();
          if (denseInt == nullptr) {
            break;
          }
          auto dataType = denseInt.getElementType().dyn_cast<IntegerType>();
          // is int32
          if (dataType == nullptr || dataType.isUnsigned() ||
              dataType.getWidth() != 32) {
            break;
          }
          llvm::SmallVector<int32_t> shape =
              llvm::to_vector(denseInt.getValues<int32_t>());
          int cntDynamic = llvm::count_if(shape, [](int32_t dimSize) {
            return dimSize == ShapedType::kDynamicSize;
          });
          if (cntDynamic == 1) {
            const ShapeValueLattice *product = operands[0];
            Attribute productAttr = product->getValue().getConstantValue();
            if (auto num = productAttr.dyn_cast_or_null<IntegerAttr>()) {
              int64_t number = num.getInt();
              if (number < 0) {
                break;
              }
              int32_t index = -1;
              for (auto elem : llvm::enumerate(shape)) {
                if (ShapedType::isDynamic(elem.value())) {
                  index = elem.index();
                } else {
                  number /= elem.value();
                }
              }
              shape[index] = number;
              attr = DenseIntElementsAttr::get(denseInt.getType(), shape);
            }
          }
        } while (0);

        LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
        propagateIfChanged(lattice, lattice->join(mlir::dataflow::ConstantValue(
                                        attr, op->getDialect())));
      })
      .Default([&](Operation *op) {
        ShapeValueAnalysis::visitOperation(op, operands, results);
      });
}
} // namespace mlir
