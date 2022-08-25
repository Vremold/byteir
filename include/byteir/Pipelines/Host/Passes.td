//===- Passes.td - Pipelines pass definition file --------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOST_PASSES
#define BYTEIR_PIPELINES_HOST_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Host Opt Pipeline
//===----------------------------------------------------------------------===//

def HostOptPipeline : Pass<"host-opt", "mlir::ModuleOp"> {
  let summary = "Host Opt Pipeline";
  let constructor = "mlir::createHostOptPipelinePass()";
  let dependentDialects = [];
  let options = [
    Option<"fileName", "file-name", "std::string",
           /*default=*/"\"host_kernels.ll\"",
           "To specify where the generated llvm kernel will be writed to">,
  ];
}

def ToLLVMPipeline : Pass<"to-llvm", "mlir::ModuleOp"> {
  let summary = "To LLVM dialect Pipeline";
  let constructor = "mlir::createToLLVMPipelinePass()";
  let dependentDialects = [
    "mlir::arith::ArithmeticDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif // BYTEIR_PIPELINES_HOST_PASSES