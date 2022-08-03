//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_MEMREF_PASSES
#define BYTEIR_MEMREF_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Apply MemRef Affine Layout
//===----------------------------------------------------------------------===//

def ApplyMemRefAffineLayout : Pass<"apply-memref-affine-layout", "func::FuncOp"> {
  let summary = "Apply applicable MemRef layout to affineMap";
  let constructor = "mlir::createApplyMemRefAffineLayoutPass()";
}


//===----------------------------------------------------------------------===//
// Reify Alloc (deprecated, using canonicalize instead) 
//===----------------------------------------------------------------------===//

def ReifyAlloc : Pass<"reify-alloc", "func::FuncOp"> {
  let summary = "Reify Alloc after constant folding (deprecated using canonicalize instead)";
  let constructor = "mlir::createReifyAllocPass()";
}

//===----------------------------------------------------------------------===//
// Simplify View
//===----------------------------------------------------------------------===//

def SimplifyView : Pass<"simplify-view", "func::FuncOp"> {
  let summary = "Simplify subview and view. Note it might create new affine.apply op if the "
                "offset of subview is dynamic.";
  let constructor = "mlir::createSimplifyViewPass()";
  let dependentDialects = [
    "AffineDialect"
  ];
}


#endif // BYTEIR_MEMREF_PASSES