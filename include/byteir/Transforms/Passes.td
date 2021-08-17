//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_TRANSFORMS_PASSES
#define BYTEIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CMAE (Common Memory Access Elimination)
//===----------------------------------------------------------------------===//

def CMAE : FunctionPass<"cmae"> {
  let summary = "Eliminate common memory access within a block";
  let constructor = "mlir::createCMAEPass()";
  let options = [
    Option<"skipAttr", "skip-attr", "std::string",
            /*default=*/"",
            "An optional attribute to skip this pass for specific function."
            "Use it for functions with non-restricted args.">
  ];
}

//===----------------------------------------------------------------------===//
// MemrefOpToStdCall
//===----------------------------------------------------------------------===//

def RewriteOpToStdCall : Pass<"rewrite-op-to-std-call", "ModuleOp"> {
  let summary = "Rewrite any op of memref to external std call.";
  let constructor = "mlir::createRewriteOpToStdCallPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "StandardOpsDialect"
  ];
  let options = [
    ListOption<"callTable", "call-table", "std::string",
               "Replace op with call function name: call-table=<op name>:<call name>,...",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

//===----------------------------------------------------------------------===//
// SetArgShape
//===----------------------------------------------------------------------===//

def SetArgShape: Pass<"set-arg-shape", "ModuleOp"> {
  let summary = "Set arg shape for the function specified.";
  let constructor = "mlir::createSetArgShapePass()";
  let options = [
    Option<"dim", "dim", "int", /*default=*/"0",
           "The dim to be set in all qulified args.">,
    Option<"size", "size", "int", 
            /*default=*/"1",
           "The size to be used in all qulified args for the specified dim.">,
    Option<"entryFuncName", "entry-func-name", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
    Option<"argAttrName", "arg-attr-name", "std::string",
            /*default=*/"",
            "Specify the arg(s) to be acted on if it contains the attribute name.">
  ];
}


#endif // BYTEIR_TRANSFORMS_PASSES