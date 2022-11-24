//===-- LinalgExtBase.td --------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_LINALG_EXT_BASE
#define BYTEIR_LINALG_EXT_BASE

include "mlir/IR/OpBase.td"

def LinalgExt_Dialect : Dialect {
  let name = "linalg_ext";

  let summary = "Extension for Linalg dialect";

  let cppNamespace = "::mlir::linalg_ext";
  let dependentDialects = [
    "arith::ArithDialect",
    "AffineDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // BYTEIR_LINALG_EXT_BASE