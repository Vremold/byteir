//===-- ByreOps.td - Byre dialect operation definitions ------*- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the Byre dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BYRE_OPS
#define BYRE_OPS

include "mlir/Dialect/DLTI/DLTIBase.td"
include "byteir/Dialect/Byre/ByreBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Byre Dialect operations.
//===----------------------------------------------------------------------===//

// Base class
class Byre_Op<string mnemonic, list<Trait> traits = []> :
    Op<Byre_Dialect, mnemonic, traits>;

def Byre_ComputeOp : Byre_Op<"compute",
    [HasParent<"func::FuncOp">, ByreInterface, MemRefsNormalizable]> {
  let summary = "compute operation defines a kernel in runtime";
  let description = [{ 
    Example:
    ```mlir
    %2 = byre.compute @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }

    FunctionType getType();
	
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

  }];

  // TODO change assemblyFormat to real parser and printer later
  // for more complicated support
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` type($operands)
  }];

  let hasVerifier = 1;
  //let skipDefaultBuilders = 1;
}

def Byre_CopyOp : Byre_Op<"copy",
    [HasParent<"func::FuncOp">, ByreInterface, CopyOpInterface]> {
  let summary = "copy operation";
  let description = [{ 
    Copies the data to from the source memref to the target memref .

    Usage:

    ```mlir
    byre.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                              memref<?xf32, stride_specification>
    ```
    Note %arg0 is source, %arg1 is destination.
  }];


  let arguments = (ins
    AnyStridedMemRef:$source,
    AnyStridedMemRef:$target);

  let extraClassDeclaration = [{
    Value getSource() { return source();}
    Value getTarget() { return target(); }
  }];

  let assemblyFormat = [{
    `(` $source `,` $target `)` attr-dict `:`
        type($source) `,` type($target)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Byre_AliasOp
    : Byre_Op<"alias",
              [HasParent<"func::FuncOp">,
               DeclareOpInterfaceMethods<ByreInterface, ["getCalleeName"]>,
               DeclareOpInterfaceMethods<ViewLikeOpInterface>,
               NoSideEffect]> {
  let summary = "alias operation";

  let arguments = (ins
    AnyStridedMemRef:$source,
    I64Attr:$offset
  );

  let results = (outs
    AnyStridedMemRef:$target
  );

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif // BYRE_OPS
