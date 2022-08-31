//===- DynamicReshape.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

LogicalResult InsertReshapeShapeConstraints(Operation *op, OpBuilder &builder) {
  builder.setInsertionPointAfter(op);
  SmallVector<Value> dimOfOperand, dimOfResult;
  auto operand = op->getOperand(0);
  auto result = op->getResult(0);
  auto oprRankedTensor = operand.getType().dyn_cast<RankedTensorType>();
  auto resRankedTensor = result.getType().dyn_cast<RankedTensorType>();
  if (!oprRankedTensor || !resRankedTensor)
    return failure();
  auto inputShape = oprRankedTensor.getShape();
  auto outputShape = resRankedTensor.getShape();
  if (inputShape.size() == 0)
    return failure();

  for (size_t i = 0; i < inputShape.size(); ++i)
    dimOfOperand.push_back(
        builder.create<tensor::DimOp>(op->getLoc(), operand, i));
  for (size_t i = 0; i < outputShape.size(); ++i)
    dimOfResult.push_back(
        builder.create<tensor::DimOp>(op->getLoc(), result, i));

  Value opr_size = dimOfOperand[0];
  for (size_t i = 1; i < dimOfOperand.size(); ++i)
    opr_size =
        builder.create<shape::MulOp>(op->getLoc(), opr_size, dimOfOperand[i]);
  Value res_size = dimOfResult[0];
  for (size_t i = 1; i < dimOfResult.size(); ++i)
    res_size =
        builder.create<shape::MulOp>(op->getLoc(), res_size, dimOfResult[i]);
  builder.create<shape_ext::MeetOp>(op->getLoc(), opr_size, res_size);

  return success();
};

void mlir::registerReshapeShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::ReshapeOp::getOperationName(), InsertReshapeShapeConstraints);
}

void mlir::registerDynamicReshapeShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::DynamicReshapeOp::getOperationName(),
      InsertReshapeShapeConstraints);
}

void mlir::registerDynamicReshapeInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      mhlo::DynamicReshapeOp::getOperationName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mlir::ShapeAdaptor shapeAdaptor = operands.getValueAsShape(1);
        if (!shapeAdaptor)
          return failure();

        ShapedTypeComponents resShape;
        shapeAdaptor.getDims(resShape);
        inferredReturnTypes.push_back(resShape);
        return success();
      });
}
