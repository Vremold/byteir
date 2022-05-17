//===-- LaceOps.td - Lace dialect operation definitions --*- tablegen -*---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef LACE_OPS
#define LACE_OPS

include "byteir/Dialect/Lace/LaceBase.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Lace Dialect operations.
//===----------------------------------------------------------------------===//

class Lace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lace_Dialect, mnemonic, traits> {
  let verifier = [{ return Verify(*this); }];
}

def Lace_ReshapeOp: Lace_Op<"reshape", [ViewLikeOpInterface, Lace_AliasLikeOpInterface]> {
  let arguments = (ins AnyStridedMemRef:$source);

  let results = (outs AnyStridedMemRef:$target);

  let extraClassDeclaration = [{
    Value getViewSource() { return source(); }
    int64_t getOffsetElem() { return 0; }
  }];
}

def Lace_SliceOp: Lace_Op<"slice", [ViewLikeOpInterface, Lace_AliasLikeOpInterface]> {
  let arguments = (ins
    AnyStridedMemRef:$source,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides);

  let results = (outs AnyStridedMemRef:$target);

  let extraClassDeclaration = [{
    Value getViewSource() { return source(); }

    int64_t getOffsetElem();

    static bool isValid(MemRefType source,
                        MemRefType target,
                        ArrayRef<int64_t> startIndices,
                        ArrayRef<int64_t> limitIndices,
                        ArrayRef<int64_t> strides);
  }];
}

#endif // LACE_OPS
