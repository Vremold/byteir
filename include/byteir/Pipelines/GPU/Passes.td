//===- Passes.td - Pipelines pass definition file --------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_PIPELINES_GPU_PASSES
#define BYTEIR_PIPELINES_GPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GPU Opt Pipeline
//===----------------------------------------------------------------------===//

def GPUOptPipeline : Pass<"gpu-opt", "mlir::ModuleOp"> {
  let summary = "GPU Opt Pipeline";
  let constructor = "mlir::createGPUOptPipelinePass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to speicify target.">,
  ];
}

//===----------------------------------------------------------------------===//
// Mamtmul Epilogue GPU Pipeline
//===----------------------------------------------------------------------===//

def MatmulEpilogueGPUPipeline : Pass<"matmul-epilogue-gpu", "mlir::ModuleOp"> {
  let summary = "Mamtmul Epilogue for gpu";
  let constructor = "mlir::createMatmulEpilogueGPUPipelinePass()";
    let dependentDialects = [
    "AffineDialect",
    "arith::ArithmeticDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to speicify target.">,
  ];
}

//===----------------------------------------------------------------------===//
// Linalg Memref GPU Pipeline
//===----------------------------------------------------------------------===//

def LinalgMemrefGPUPipeline : Pass<"linalg-memref-gpu", "mlir::ModuleOp"> {
  let summary = "Linalg Opt Pipeline in Memref for GPU";
  let constructor = "mlir::createLinalgMemrefGPUPipelinePass()";
  let options = [
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to speicify target.">,
  ];
}

//===----------------------------------------------------------------------===//
// NVVM Codegen Pipeline
//===----------------------------------------------------------------------===//

def NVVMCodegenPipeline : Pass<"nvvm-codegen", "mlir::ModuleOp"> {
  let summary = "NVVM Codegen Pipeline";
  let constructor = "mlir::createNVVMCodegenPipelinePass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::cf::ControlFlowDialect",   
    "mlir::arith::ArithmeticDialect",
    "mlir::func::FuncDialect",
  ];
}

#endif // BYTEIR_PIPELINES_GPU_PASSES