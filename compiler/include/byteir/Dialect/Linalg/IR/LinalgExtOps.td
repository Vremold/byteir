//===-- LinalgExtOps.td ---------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_LINALG_EXT_OPS
#define BYTEIR_LINALG_EXT_OPS

include "byteir/Dialect/Linalg/IR/LinalgExtBase.td"
include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for the operation in this dialect
class LinalgExt_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    LinalgExt_BaseOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DestinationStyleOpInterface,
         LinalgExtInterface,
  ])> {

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  code extraLinalgExtOpClassDeclaration = [{
    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      std::pair<unsigned, unsigned> outputsIndexAndLength =
        getODSOperandIndexAndLength(1);
      return std::make_pair<int64_t, int64_t>(
          outputsIndexAndLength.first,
          outputsIndexAndLength.first + outputsIndexAndLength.second);
    }

  }];
}

//===----------------------------------------------------------------------===//
// Basic ops
//===----------------------------------------------------------------------===//

def LinalgExt_YieldOp : LinalgExt_BaseOp<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def LinalgExt_AliasOp : LinalgExt_BaseOp<"alias", [Pure]> {
  let summary = "LinalgExt alias op";
  let description = [{
    `linalg_ext.alias` is a special op  to represent alias in basic block
    It will be resolved in a special rewrite pattern. 
    It won't be removed in the regular canonicalizer.
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$operand), [{
      build($_builder, $_state, operand.getType(), operand);
    }]> 
  ];

  let hasVerifier = 1;

  let assemblyFormat = " `(` $operand `:` type($operand) `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Advanced ops
//===----------------------------------------------------------------------===//

def LinalgExt_SoftmaxOp : LinalgExt_Op<"softmax",
    [DeclareOpInterfaceMethods<LinalgExtInterface, 
        ["isValidTiling",
         "correctTiledConsumerOps"]>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition", 
         "getTiledImplementation"]>]> {

  let summary = "Softmax Op";
  let description = [{
    Computes softmax along a given dimension.
    Tensor:
      [result, max, accum, scale] = softmax {dimension} ins(data) outs(result, max, accum, scale);
    Memeref
      softmax {dimension} ins(data) outs(result, scale, max, accum)
    Computation: 
      max_new = max(max_old, max(data along dimension)).
      accum_new = accum_old * exp(max_old - max_new) + sum(exp(data - max_new)). 
      scale = accum_old * exp(max_old - max_new) / accum_new.
      resutl = exp(data - max_new)/ sum(exp(data - max_new)).
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    `dimension` `(` $dimension `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let hasFolder = 1;

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    Value max() {
      return getOutputOperand(1)->get();
    }
    Value accumulator() {
      return getOutputOperand(2)->get();
    }
    Value scale() {
      return getOutputOperand(3)->get();
    }
    ShapedType getOperandType() {
      return output().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];

}


def LinalgExt_DiagOp : LinalgExt_Op<"diag", 
    [DeclareOpInterfaceMethods<LinalgExtInterface>]> {

  let summary = "Diag Op";
  let description = [{
    diag(x) presents a diag matrix from a vector.
    it is an intermediate op for softmax's fusion
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    attr-dict 
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$output), [{
      build($_builder, $_state, {output.getType()}, {input}, {output});
    }]> 
  ];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return output().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }

    static Type getDiagType(ShapedType type);
  }];

}

def LinalgExt_ScanOp : LinalgExt_Op<"scan",
    [DeclareOpInterfaceMethods<LinalgExtInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["generateResultTileValue",
       "getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Scan operator";
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension, CArg<"bool", "true">:$inclusive)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value accumulator() {
      return getOutputOperand(1)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];
}

def LinalgExt_CustomOp : LinalgExt_Op<"custom",
    [DeclareOpInterfaceMethods<LinalgExtInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
        ["generateResultTileValue",
         "getIterationDomain",
         "getLoopIteratorTypes",
         "getResultTilePosition", 
         "getTiledImplementation"]>,
      SingleBlockImplicitTerminator<"::mlir::linalg_ext::YieldOp">,
    ]> {

  let summary = "custom op wrapper";
  let description = [{
    Custom op wrapper. TODO add an exmaple here.
  }];

  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
   let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    attr-dict 
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
  }];
}

#endif // BYTEIR_LINALG_EXT_OPS