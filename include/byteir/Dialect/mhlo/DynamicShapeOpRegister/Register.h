//===- Register.h ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MHLO_DYNAMIC_SHAPE_OP_REGISTER_H
#define BYTEIR_DIALECT_MHLO_DYNAMIC_SHAPE_OP_REGISTER_H

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// StaticShapeInfer Registration
//===----------------------------------------------------------------------===//
void registerConvolutionInferReturnTypeComponents();
void registerDotGeneralInferReturnTypeComponents();
void registerDynamicBroadcastInDimInferReturnTypeComponents();
void registerDynamicReshapeInferReturnTypeComponents();
void registerRealDynamicSliceInferReturnTypeComponents();
void registerReduceInferReturnTypeComponents();
void registerSoftmaxInferReturnTypeComponents();
void registerTorchIndexSelectInferReturnTypeComponents();

inline void registerAllMhloInferReturnTypeComponents() {
  registerConvolutionInferReturnTypeComponents();
  registerDotGeneralInferReturnTypeComponents();
  registerDynamicBroadcastInDimInferReturnTypeComponents();
  registerDynamicReshapeInferReturnTypeComponents();
  registerRealDynamicSliceInferReturnTypeComponents();
  registerReduceInferReturnTypeComponents();
  registerSoftmaxInferReturnTypeComponents();
  registerTorchIndexSelectInferReturnTypeComponents();
}

//===----------------------------------------------------------------------===//
// BoundedShapeInfer Registration
//===----------------------------------------------------------------------===//
void registerDynamicPartitionInferBoundedReturnTypeComponents();
void registerNonZeroInferBoundedReturnTypeComponents();
void registerWhereInferBoundedReturnTypeComponents();

inline void registerAllMhloInferBoundedReturnTypeComponents() {
  registerDynamicPartitionInferBoundedReturnTypeComponents();
  registerNonZeroInferBoundedReturnTypeComponents();
  registerWhereInferBoundedReturnTypeComponents();
}

//===----------------------------------------------------------------------===//
// ShapeReification Registration
//===----------------------------------------------------------------------===//
void registerDotReifyReturnTypeShapes();
void registerDynamicStitchReifyReturnTypeShapes();
void registerDynamicMaskStitchReifyReturnTypeShapes();
void registerDynamicBroadcastInDimReifyReturnTypeShapes();
void registerSoftmaxReifyReturnTypeShapes();
void registerTorchIndexSelectReifyReturnTypeShapes();

inline void registerAllMhloReifyReturnTypeShapes() {
  registerDotReifyReturnTypeShapes();
  registerDynamicStitchReifyReturnTypeShapes();
  registerDynamicMaskStitchReifyReturnTypeShapes();
  registerDynamicBroadcastInDimReifyReturnTypeShapes();
  registerSoftmaxReifyReturnTypeShapes();
  registerTorchIndexSelectReifyReturnTypeShapes();
}

//===----------------------------------------------------------------------===//
// ShapeConstraint Registration
//===----------------------------------------------------------------------===//
void registerDotGeneralShapeConstraints();
void registerDynamicPartitionShapeConstraints();
void registerDynamicReshapeShapeConstraints();
void registerEinsumShapeConstraints();
void registerReshapeShapeConstraints();

inline void registerAllMhloShapeConstraints() {
  registerDotGeneralShapeConstraints();
  registerDynamicPartitionShapeConstraints();
  registerDynamicReshapeShapeConstraints();
  registerEinsumShapeConstraints();
  registerReshapeShapeConstraints();
}

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_DYNAMIC_SHAPE_OP_REGISTER_H
