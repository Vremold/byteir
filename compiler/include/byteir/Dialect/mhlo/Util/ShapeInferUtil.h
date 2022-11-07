//===- ShapeInferUtil.h ---------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

constexpr StringRef getBoundedShapeDenseAttrName() {
  return "byteir.bounded_shape_dense";
}
//===----------------------------------------------------------------------===//
// runShapeInference
//===----------------------------------------------------------------------===//

// This could be used for both bounded shape inference and static shape
// inference. Return failure if a shape mismatch occurs.
LogicalResult runShapeInference(func::FuncOp funcOp,
                                bool isBoundedShapeInfer = false);

//===----------------------------------------------------------------------===//
// ReifyReturnTypeShapes Registration
//===----------------------------------------------------------------------===//

// The function signature is similar to reifyReturnTypeShapes's, except that
// it has an additional argument of type `Operation *`. It should be easy if
// we decice to contribute some of the implementation to upstream later.
using ReifyReturnTypeShapes = std::function<LogicalResult(
    Operation *op, OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<::mlir::Value> &reifiedReturnShapes)>;

struct ReifyReturnTypeShapesRegistration {
  ReifyReturnTypeShapesRegistration(llvm::StringRef name,
                                    const ReifyReturnTypeShapes &function);
};

ReifyReturnTypeShapes reifyReturnTypeShapes(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InsertShapeConstraint Registration
//===----------------------------------------------------------------------===//

using InsertShapeConstraint =
    std::function<LogicalResult(Operation *op, OpBuilder &builder)>;

struct InsertShapeConstraintRegistration {
  InsertShapeConstraintRegistration(llvm::StringRef name,
                                    const InsertShapeConstraint &function);
};

InsertShapeConstraint insertShapeConstraint(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InferBoundedReturnTypeComponents Registration
//===----------------------------------------------------------------------===//

using InferBoundedReturnTypeComponents = std::function<LogicalResult(
    MLIRContext *, Optional<Location>, ValueShapeRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes)>;

struct InferBoundedReturnTypeComponentsRegistration {
  InferBoundedReturnTypeComponentsRegistration(
      llvm::StringRef name, const InferBoundedReturnTypeComponents &function);
};

InferBoundedReturnTypeComponents
inferBoundedReturnTypeComponents(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InferReturnTypeComponents Registration, for static-shape-infer
//===----------------------------------------------------------------------===//

using InferReturnTypeComponents = std::function<LogicalResult(
    MLIRContext *, Optional<Location>, ValueShapeRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes)>;

struct InferReturnTypeComponentsRegistration {
  InferReturnTypeComponentsRegistration(
      llvm::StringRef name, const InferReturnTypeComponents &function);
};

InferReturnTypeComponents inferReturnTypeComponents(llvm::StringRef name);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H
