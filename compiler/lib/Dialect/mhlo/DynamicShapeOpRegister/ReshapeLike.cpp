//===- ReshapeLike.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

  Value opr_size;
  if (dimOfOperand.size() == 0) {
    opr_size = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  } else {
    opr_size = dimOfOperand[0];
    for (size_t i = 1; i < dimOfOperand.size(); ++i)
      opr_size =
          builder.create<shape::MulOp>(op->getLoc(), opr_size, dimOfOperand[i]);
  }

  Value res_size;
  if (dimOfResult.size() == 0) {
    res_size = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  } else {
    res_size = dimOfResult[0];
    for (size_t i = 1; i < dimOfResult.size(); ++i)
      res_size =
          builder.create<shape::MulOp>(op->getLoc(), res_size, dimOfResult[i]);
  }
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
