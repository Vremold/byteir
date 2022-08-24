//===- NonZero.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

/// See NonZero's signature on
/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonzero
void mlir::registerNonZeroInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getNonZeroName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        Value input = operands[0];
        ShapedType inputShape = input.getType().dyn_cast<ShapedType>();
        if (!inputShape || !inputShape.hasStaticShape())
          return failure();

        Type type = RankedTensorType::get({inputShape.getNumElements()},
                                          IntegerType::get(context, 64));
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}