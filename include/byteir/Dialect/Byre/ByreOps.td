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

#ifndef Byre_OPS
#define Byre_OPS

include "mlir/Dialect/DLTI/DLTIBase.td"
include "byteir/Dialect/Byre/ByreBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Byre Dialect operations.
//===----------------------------------------------------------------------===//

// Base class
class Byre_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Byre_Dialect, mnemonic, traits>;

def Byre_ComputeOp : Byre_Op<"compute",
    [HasParent<"FuncOp">, MemRefsNormalizable]> {
  let summary = "compute operation defines a kernel in runtime";
  let description = [{ 
    Example:
    ```mlir
    %2 = byre.compute @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  //let results = (outs Variadic<AnyType>);

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

  let verifier = [{ return ::verify(*this); }];
  //let skipDefaultBuilders = 1;
}

def Byre_CopyOp : Byre_Op<"copy",
    [HasParent<"FuncOp">, CopyOpInterface]> {
  let summary = "copy operation";
  let description = [{ TODO
    Copies the data to the output view from the input view.

    Usage:

    ```mlir
    byre.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                              memref<?xf32, stride_specification>
    ```
    Note %arg0 is destination, %arg1 is source.
  }];


  let arguments = (ins
    AnyStridedMemRef:$output,
    AnyStridedMemRef:$input);

  let builders = [
    OpBuilder<(ins "Value":$output, "Value":$input,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    ValueRange outputs() { return getOperands().take_front(); }
    ValueRange inputs() { return getOperands().take_back(); }

    Value getSource() { return input();}
    Value getTarget() { return output(); }

  }];

  let assemblyFormat = [{
    `(` $input `,` $output `)` attr-dict `:`
        type($input) `,` type($output)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let skipDefaultBuilders = 1;
  let verifier = [{ return ::verify(*this); }];
}

#endif // BYRE_OPS
