//===- TorchIndexSelect.cpp -----------------------------------*--- C++ -*-===//
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

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

/// This is in fact a normal shape-infer function for mhlo.torch_index_select.
/// It is not specially designed for bounded shape infer, we just need a shape
/// infer func in the bounded-shape-infer pass
/// FIXME: remove if upstream implement
void mlir::registerTorchIndexSelectInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      mhlo::TorchIndexSelectOp::getOperationName(),
      [](MLIRContext *context, Optional<Location> loc, ValueShapeRange operands,
         DictionaryAttr attrs, RegionRange regions,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mhlo::TorchIndexSelectOp::Adaptor adaptor(operands, attrs, regions);
        uint64_t batchDims = adaptor.batch_dims();
        uint64_t dim = adaptor.dim();

        SmallVector<int64_t> dataShape;
        SmallVector<int64_t> indicesShape;
        operands.getShape(0).getDims(dataShape);
        operands.getShape(1).getDims(indicesShape);

        SmallVector<int64_t> resultShape;
        // check dim size match, set `dim` to the static shape if possible
        auto dimMatch = [](int64_t x, int64_t y, int64_t *dim) {
          if (ShapedType::isDynamic(x)) {
            *dim = y;
            return true;
          }
          if (ShapedType::isDynamic(y)) {
            *dim = x;
            return true;
          }
          *dim = x;
          return x == y;
        };

        for (uint64_t i = 0, e = batchDims; i < e; ++i) {
          int64_t dimSize;
          if (!dimMatch(dataShape[i], indicesShape[i], &dimSize)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "data and indices batch_dims mismatch at dim #" << i
                       << ": " << dataShape[i] << " != " << indicesShape[i]
                       << "\n");
            return failure();
          }
          resultShape.push_back(dimSize);
        }

        for (uint64_t i = batchDims, e = dim; i < e; ++i) {
          resultShape.push_back(dataShape[i]);
        }
        for (uint64_t i = batchDims, e = indicesShape.size(); i < e; ++i) {
          resultShape.push_back(indicesShape[i]);
        }
        for (uint64_t i = dim + 1, e = dataShape.size(); i < e; ++i) {
          resultShape.push_back(dataShape[i]);
        }

        if (auto dataType =
                operands.getTypes()[0].dyn_cast_or_null<ShapedType>()) {
          auto outElement = dataType.getElementType();
          Type retType = RankedTensorType::get(resultShape, outElement);
          inferredReturnTypes.push_back(retType.cast<ShapedType>());
          return success();
        }
        LLVM_DEBUG(llvm::dbgs() << loc << " get output element type failed\n");
        return failure();
      });
}
