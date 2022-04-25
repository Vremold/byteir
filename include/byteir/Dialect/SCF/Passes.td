//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_SCF_PASSES
#define BYTEIR_SCF_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InsertTrivialSCFLoop
//===----------------------------------------------------------------------===//

def InsertTrivialSCFLoop : Pass<"insert-trivial-scf-loop", "mlir::FuncOp"> {
  let summary = "Insert trivial scf Loop for scalar computation";
  let constructor = "mlir::createInsertTrivialSCFLoopPass()";
  let dependentDialects = [
    "scf::SCFDialect"
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

#endif // BYTEIR_SCF_PASSES