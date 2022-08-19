
//===- RealDynamicSlice.cpp -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bounded-shape-infer"

using namespace mlir;

/// TODO: push to upstream
void mlir::registerRealDynamicSliceInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      mhlo::RealDynamicSliceOp::getOperationName(),
      [](MLIRContext *context, Optional<Location> loc, ValueShapeRange operands,
         DictionaryAttr attrs, RegionRange regions,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mhlo::RealDynamicSliceOp::Adaptor adaptor(operands, attrs, regions);

        // here `getValueAsShape` just get the constant values, they're
        // in fact not shapes, but the start,limit,stride indices of slice.
        ShapeAdaptor startIndicesAdaptor = operands.getValueAsShape(1);
        ShapeAdaptor limitIndicesAdaptor = operands.getValueAsShape(2);
        ShapeAdaptor stridesAdaptor = operands.getValueAsShape(3);
        if (!startIndicesAdaptor || !limitIndicesAdaptor || !stridesAdaptor) {
          LLVM_DEBUG(llvm::dbgs()
                     << loc << " start,strides,limit not all constant\n");
          return failure();
        }
        SmallVector<int64_t> startIndices;
        SmallVector<int64_t> limitIndices;
        SmallVector<int64_t> strides;

        startIndicesAdaptor.getDims(startIndices);
        limitIndicesAdaptor.getDims(limitIndices);
        stridesAdaptor.getDims(strides);

        int64_t rank = startIndices.size();

        if (!(rank == limitIndices.size() && rank == strides.size())) {
          LLVM_DEBUG(llvm::dbgs() << "start,limit,strides rank mismatch");
          return failure();
        }

        SmallVector<int64_t> dimensions(rank);
        for (int64_t i = 0; i < rank; ++i)
          dimensions[i] =
              (limitIndices[i] - startIndices[i] + strides[i] - 1) / strides[i];

        if (auto inputType = operands[0].getType().dyn_cast<ShapedType>()) {
          auto outElement = inputType.getElementType();
          Type retType = RankedTensorType::get(dimensions, outElement);
          inferredReturnTypes.push_back(retType.cast<ShapedType>());
          return success();
        }

        return failure();
      });
}
