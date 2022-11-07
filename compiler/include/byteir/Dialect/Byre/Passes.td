//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_BYRE_PASSES
#define BYTEIR_BYRE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ByreFold
//===----------------------------------------------------------------------===//

def ByreFold : Pass<"byre-fold", "func::FuncOp"> {
  let summary = "Fold Byre Ops";
  let constructor = "mlir::createByreFoldPass()";
}

#endif // BYTEIR_BYRE_PASSES