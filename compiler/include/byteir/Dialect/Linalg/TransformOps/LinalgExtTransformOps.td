//===-- LinalgExtTransformOps.td ------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_LINALG_EXT_TRANSFORMOPS
#define BYTEIR_LINALG_EXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


def AnnotateOp : Op<Transform_Dialect, "annotate",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is for annotation.
  }];

  let arguments = (ins PDL_Operation:$target);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
  }];
}

def FuseExtOp : Op<Transform_Dialect, "structured.fuse_ext",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Tiles the operations pointed to by the target handle and fuses their
    producers greedily using the options provided as attributes.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def TileLabelOp : Op<Transform_Dialect, "structured.tile_label",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is a tile annotation
  }];

  let arguments = (ins PDL_Operation:$target);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
  }];
}

def TileExtOp : Op<Transform_Dialect, "structured.tile_ext",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is a tile extension
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

#endif // BYTEIR_LINALG_EXT_TRANSFORMOPS