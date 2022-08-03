//===-- ShapeExtBase.td ---------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_SHAPE_EXT_BASE
#define BYTEIR_SHAPE_EXT_BASE

include "mlir/Dialect/Shape/IR/ShapeBase.td"

def ShapeExtDialect : Dialect {
  let name = "shape_ext";

  let summary = "Extension for shape dialect";

  let cppNamespace = "::mlir::shape_ext";
  let dependentDialects = ["arith::ArithmeticDialect", "tensor::TensorDialect", "shape::ShapeDialect"];

  // let useDefaultTypePrinterParser = 1;
  // let hasConstantMaterializer = 1;
  // let hasOperationAttrVerify = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // BYTEIR_SHAPE_EXT_BASE