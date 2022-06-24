//===-- ShapeExtOps.td -------------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_SHAPE_EXT_OPS
#define BYTEIR_SHAPE_EXT_OPS

include "byteir/Dialect/Shape/ShapeExtBase.td"


// Base class for the operation in this dialect
class ShapeExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<ShapeExtDialect, mnemonic, traits>;

def ShapeExt_TieOp: ShapeExt_Op<"tie"> {
  let summary = "binding a set of multidim shape.size with a tensor value";
  let description = [{
    Binding a set of multidim shape.size with a tensor value.
  }];
  let arguments = (ins
    AnyTensor:$value,
    Arg<Variadic<Shape_SizeOrIndexType>, "">:$dims
  );
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}


def ShapeExt_MeetOp: ShapeExt_Op<"meet"> {
  let summary = "an alternative of shape.meet to support index type";
  let arguments = (ins
    Shape_SizeOrIndexType:$arg0,
    Shape_SizeOrIndexType:$arg1
  );
}

#endif // BYTEIR_SHAPE_EXT_OPS