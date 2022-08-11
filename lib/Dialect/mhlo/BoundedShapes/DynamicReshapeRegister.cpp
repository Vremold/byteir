//===- DynamicReshapeRegister.h -------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bounded-shape-infer"

using namespace mlir;

void mlir::registerDynamicReshapeInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
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
