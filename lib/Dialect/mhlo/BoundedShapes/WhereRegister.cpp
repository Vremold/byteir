//===- WhereRegister.h ----------------------------------------*--- C++ -*-===//
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

/// See Where's signature on https://www.tensorflow.org/api_docs/python/tf/where
/// Bounded shape infer is the same as nonzero
void mlir::registerWhereInferBoundedReturnTypes() {
  static InferBoundedReturnTypesRegistration shapeRegister(
      getWhereName(), [](MLIRContext *context, Optional<Location>,
                         ValueRange operands, DictionaryAttr, RegionRange,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
        Value input = operands[0];
        ShapedType inputShape = input.getType().dyn_cast<ShapedType>();
        if (!inputShape || !inputShape.hasStaticShape())
          return failure();

        inferredReturnTypes.push_back(RankedTensorType::get(
            {inputShape.getNumElements(), inputShape.getRank()},
            IntegerType::get(context, 64)));
        return success();
      });
}
