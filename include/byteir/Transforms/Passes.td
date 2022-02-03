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

def CMAE : Pass<"cmae", "mlir::FuncOp"> {
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
// CollectFunc
//===----------------------------------------------------------------------===//

def CollectFunc : Pass<"collect-func", "ModuleOp"> {
  let summary = "collect all function based on anchor";
  let constructor = "mlir::createCollectFuncPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">
  ];
}

//===----------------------------------------------------------------------===//
// FuncTag
//===----------------------------------------------------------------------===//

def FuncTag : Pass<"func-tag", "ModuleOp"> {
  let summary = "Attach Tag based on func name";
  let constructor = "mlir::createFuncTagPass()";
  let options = [
    Option<"attachAttr", "attach-attr", "std::string",
            /*default=*/"",
            "An optional attribute attaching on target functions: <attr_name>:<type>:<value>">,
    Option<"funcName", "func-name", "std::string",
            /*default=*/"",
            "An optional name to specify target functions.">
  ];
}

//===----------------------------------------------------------------------===//
// GraphClusteringByDevice
//===----------------------------------------------------------------------===//
// This pass is currently a naive one and only works correct on tensor level Dialects
def GraphClusteringByDevice : Pass<"graph-clustering-by-device", "ModuleOp"> {
  let summary = "Clustering each function by device.";
  let constructor = "mlir::createGraphClusteringByDevicePass()";
  let options = [
    Option<"attrName", "attr-name", "std::string", /*default=*/"\"device\"",
           "The attr name used for clustering.">,
    Option<"device", "device", "std::string", /*default=*/"\"device\"",
           "The concrete device name for newly created device functions.">
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
// LoopUnroll
//===----------------------------------------------------------------------===//

def LoopUnroll : Pass<"unroll", "mlir::FuncOp"> {
  let summary = "Unroll scf/affine loops for anchored loop or satisfying depth";
  let constructor = "mlir::createByteIRLoopUnrollPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"2",
           "Use this unroll factor for all loops being unrolled">,
    Option<"unrollUpToFactor", "unroll-up-to-factor", "bool", /*default=*/"false", 
           "Allow unrolling up to the factor specified">,
    Option<"unrollFull", "unroll-full", "bool", /*default=*/"false",
           "Fully unroll loops">,
    Option<"depth", "depth", "int", /*default=*/"-1",
           "Loop depth, a negative number implies disabling">,
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