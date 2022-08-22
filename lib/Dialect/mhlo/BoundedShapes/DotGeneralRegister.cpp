//===- DotGeneralRegister.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bounded-shape-infer"

using namespace mlir;

/// TODO: push to upstream
void mlir::registerDotGeneralInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      mhlo::DotGeneralOp::getOperationName(),
      [](MLIRContext *context, Optional<Location> loc, ValueShapeRange operands,
         DictionaryAttr attrs, RegionRange regions,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mhlo::DotGeneralOp::Adaptor adaptor(operands, attrs, regions);
        auto lhsType = operands.getTypes()[0].dyn_cast<ShapedType>();
        auto rhsType = operands.getTypes()[1].dyn_cast<ShapedType>();
        if (!lhsType || !rhsType)
          return failure();

        // get shape of operands
        SmallVector<int64_t> lhsShape;
        SmallVector<int64_t> rhsShape;
        operands.getShape(0).getDims(lhsShape);
        operands.getShape(1).getDims(rhsShape);

        auto dimNumbers = adaptor.dot_dimension_numbers();
        SmallVector<int64_t> dimensions;
        for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
          dimensions.push_back(lhsShape[lhsDim]);
        }

        for (int64_t i = 0, e = lhsShape.size(); i < e; ++i) {
          if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
            dimensions.push_back(lhsShape[i]);
          }
        }

        for (int64_t i = 0, e = rhsShape.size(); i < e; ++i) {
          if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
            dimensions.push_back(rhsShape[i]);
          }
        }

        auto outElement = lhsType.getElementType();
        Type retType = RankedTensorType::get(dimensions, outElement);
        inferredReturnTypes.push_back(retType.cast<ShapedType>());

        return success();
      });
}
