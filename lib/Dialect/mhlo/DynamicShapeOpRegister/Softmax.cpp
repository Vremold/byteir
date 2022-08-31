//===- Softmax.cpp --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerSoftmaxReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getSoftmaxName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        Value dataShape =
            builder.create<shape::ShapeOfOp>(op->getLoc(), operands[0]);
        reifiedReturnShapes.push_back(dataShape);
        return success();
      });
}

void mlir::registerSoftmaxInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getSoftmaxName(),
      [](MLIRContext *context, Optional<Location> loc, ValueShapeRange operands,
         DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        ShapedType dataType = operands[0].getType().dyn_cast<ShapedType>();
        if (!dataType) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": get dataType failed\n");
          return failure();
        }
        SmallVector<int64_t> dataShape;
        operands.getShape(0).getDims(dataShape);
        inferredReturnTypes.emplace_back(dataShape, dataType.getElementType());
        return success();
      });
}
