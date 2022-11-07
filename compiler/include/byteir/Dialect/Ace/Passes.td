//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ACE_PASSES
#define BYTEIR_ACE_PASSES

include "mlir/Pass/PassBase.td"

def AceBufferize : Pass<"ace-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the `ace` dialect";
  let constructor = "mlir::createAceBufferizePass()";
}

#endif // BYTEIR_ACE_PASSES
