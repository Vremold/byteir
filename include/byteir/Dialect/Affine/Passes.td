//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_AFFINE_PASSES
#define BYTEIR_AFFINE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RewriteAffineToMemref
//===----------------------------------------------------------------------===//

def RewriteAffineToMemref : Pass<"rewrite-affine-to-memref", "mlir::FuncOp"> {
  let summary = "Rewrite Affine Load/Store to Memref Load/Store";
  let constructor = "mlir::createRewriteAffineToMemrefPass()";
  let dependentDialects = [
    "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AffineLoopFusionEx
//===----------------------------------------------------------------------===//

def AffineLoopFusionEx : Pass<"affine-loop-fusion-ex", "mlir::FuncOp"> {
  let summary = "Affine Loop Fusion Extension";
  let constructor = "mlir::createAffineLoopFusionExPass()";
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

#endif // BYTEIR_AFFINE_PASSES