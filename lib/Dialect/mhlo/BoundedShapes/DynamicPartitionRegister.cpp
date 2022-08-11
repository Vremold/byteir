//===- DynamicPartitionRegister.h -----------------------------*--- C++ -*-===//
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

/// See DynamicPartition's signature on
/// https://www.tensorflow.org/api_docs/python/tf/dynamic_partition
void mlir::registerDynamicPartitionInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getDynamicPartitionName(),
      [](MLIRContext *context, Optional<Location>, ValueShapeRange operands,
         DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto numPartition = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                                .getAs<IntegerAttr>("num_partitions")
                                .getInt();
        if (ShapedType shapedType =
                operands[0].getType().dyn_cast_or_null<ShapedType>()) {
          inferredReturnTypes.append(numPartition, shapedType);
          return success();
        }
        return failure();
      });
}