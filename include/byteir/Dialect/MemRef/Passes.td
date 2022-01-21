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

def ApplyMemRefAffineLayout : Pass<"apply-memref-affine-layout", "mlir::FuncOp"> {
  let summary = "Apply applicable MemRef layout to affineMap";
  let constructor = "mlir::createApplyMemRefAffineLayoutPass()";
}

//===----------------------------------------------------------------------===//
// Reify Alloc
//===----------------------------------------------------------------------===//

def ReifyAlloc : Pass<"reify-alloc", "mlir::FuncOp"> {
  let summary = "Reify Alloc after constant folding";
  let constructor = "mlir::createReifyAllocPass()";
}


#endif // BYTEIR_MEMREF_PASSES