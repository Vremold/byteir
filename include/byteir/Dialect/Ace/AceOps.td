//===-- AceOps.td - Ace dialect operation definitions --*- tablegen -*---===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef ACE_OPS
#define ACE_OPS

include "byteir/Dialect/Ace/AceBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// Ace Dialect operations.
//===----------------------------------------------------------------------===//

class Ace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ace_Dialect, mnemonic, traits> {
}

def Ace_ConstOp : Ace_Op<"constant", [
    ConstantLike, NoSideEffect, AllTypesMatch<["value", "output"]>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    Ace_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  // FIXME: let this op only has generic format for golden ref calculate.
  // let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
  let hasVerifier = 0;
}


def Ace_ActivateOp : Ace_Op<"activate", [NoSideEffect]> {
  let summary = "Activate operation";

  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$act_func
  );

  let results = (
    outs AnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins "::mlir::Type":$output, "::mlir::Value":$input,
      "::mlir::StringAttr":$actType,
      "llvm::ArrayRef<NamedAttribute>":$attrs),
      [{
        $_state.addOperands(input);
        $_state.addAttribute(act_funcAttrName(odsState.name), actType);
        $_state.addTypes(output);
        $_state.addAttributes(attrs);
      }]>,
  ];
}


def Ace_ReturnOp : Ace_Op<"return", [NoSideEffect, Terminator]> {
  let summary = [{
    The `ace.return` operation terminates a region and returns values.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$results
  );
}


def Ace_OpaqueOp : Ace_Op<"opaque", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "Wraps ops that could not be lower to mhlo.";

  let description = [{ 
    Example:
    ```mlir
    %0 = "ace.opaque"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
      %3 = "tf.Add"(%arg2, %arg3) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      "ace.return"(%3) : (tensor<2x4xf32>) -> ()
    }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    ```
  }];

  let regions = (region SizedRegion<1>:$body);

  let arguments = (ins
    Variadic<AnyTensor>:$inputs
  );

  let results = (outs
    Variadic<AnyTensor>:$outputs
  );

  let extraClassDeclaration = [{
    Block* addEntryBlock();
  }];

}

def Ace_CustomCallOp : Ace_Op<"custom_call", [NoSideEffect]> {
  let summary = "CustomCall operator";

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    StrAttr:$call_target_name
  );

  let results = (outs Variadic<AnyTensor>:$results);
}


#endif // ACE_OPS
