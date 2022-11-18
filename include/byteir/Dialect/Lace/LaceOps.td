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
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Lace Dialect operations.
//===----------------------------------------------------------------------===//

class Lace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lace_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Lace alias like operations
//===----------------------------------------------------------------------===//

class Lace_AliasOp<string mnemonic, list<Trait> traits = []> : 
  Lace_Op<mnemonic, !listconcat(traits,
      [ViewLikeOpInterface, Pure, Lace_AliasLikeOpInterface])>;

def Lace_ReshapeOp: Lace_AliasOp<"reshape"> {
  let arguments = (ins AnyStridedMemRef:$source);

  let results = (outs AnyStridedMemRef:$target);

  let extraClassDeclaration = [{
    Value getViewSource() { return source(); }
    int64_t getOffsetElem() { return 0; }
  }];

  let hasVerifier = 1;
}

def Lace_SliceOp: Lace_AliasOp<"slice"> {
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

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Lace computing operations
//===----------------------------------------------------------------------===//

def Lace_ConstOp : Lace_Op<"constant"> {
  let summary = "Constant operator";

  let arguments = (ins
    ElementsAttr:$value,
    Arg<Lace_BufferType, "", [MemWrite]>:$output
  );
}

def Lace_CustomCallOp : Lace_Op<"custom_call", [AttrSizedOperandSegments]> {
  let summary = "CustomCall operator";

  let arguments = (ins
    Arg<Variadic<Lace_BufferType>, "", [MemRead]>:$inputs,
    Arg<Variadic<Lace_BufferType>, "", [MemWrite]>:$output,
    StrAttr:$call_target_name,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config
  );
}

#endif // LACE_OPS
