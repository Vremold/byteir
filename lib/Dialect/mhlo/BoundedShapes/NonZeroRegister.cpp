//===- NonZeroRegister.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

/// See NonZero's signature on
/// https://github.com/onnx/onnx/blob/main/docs/Operators.md#nonzero
void mlir::registerNonZeroInferBoundedReturnTypes() {
  static InferBoundedReturnTypesRegistration shapeRegister(
      getNonZeroName(), [](MLIRContext *context, Optional<Location>,
                           ValueRange operands, DictionaryAttr, RegionRange,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
        Value input = operands[0];
        ShapedType inputShape = input.getType().dyn_cast<ShapedType>();
        if (!inputShape || !inputShape.hasStaticShape())
          return failure();

        inferredReturnTypes.push_back(RankedTensorType::get(
            {inputShape.getNumElements()}, IntegerType::get(context, 64)));
        return success();
      });
}