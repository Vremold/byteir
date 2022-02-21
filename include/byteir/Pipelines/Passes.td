//===- Passes.td - Pipelines pass definition file --------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_PIPELINES_PASSES
#define BYTEIR_PIPELINES_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
//
// Pipeline passes of 2nd-order pipelines for specific dialects
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Affine Opt Pipeline
//===----------------------------------------------------------------------===//

def AffineOptPipeline : Pass<"affine-opt", "mlir::ModuleOp"> {
  let summary = "Affine Opt Pipeline";
  let constructor = "mlir::createAffineOptPipelinePass()";
  let dependentDialects = [
    "mlir::AffineDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Byre Opt Pipeline
//===----------------------------------------------------------------------===//

def ByreOptPipeline : Pass<"byre-opt", "mlir::ModuleOp"> {
  let summary = "Byre Opt Pipeline";
  let constructor = "mlir::createByreOptPipelinePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
  ];
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "An optional string to speicify entry function.">,
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,	
  ];
}

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
// Hlo Opt Pipeline
//===----------------------------------------------------------------------===//

def HloOptPipeline : Pass<"hlo-opt", "mlir::ModuleOp"> {
  let summary = "Hlo Opt Pipeline";
  let constructor = "mlir::createHloOptPipelinePass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "An optional string to speicify entry function.">,
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to speicify target.">,
    Option<"outlineSingleElemwiseOp", "outline-single-elemwise-op", 
           "bool", /*default=*/"false",
           "whether to outline the single element-wise operation as an independent function">,
  ];
}

//===----------------------------------------------------------------------===//
// Linalg Opt Pipeline
//===----------------------------------------------------------------------===//

def LinalgOptPipeline : Pass<"linalg-opt", "mlir::ModuleOp"> {
  let summary = "Linalg Opt Pipeline";
  let constructor = "mlir::createLinalgOptPipelinePass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
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
    "mlir::StandardOpsDialect",   
  ];
}


//===----------------------------------------------------------------------===//
// Total Bufferize Pipeline
//===----------------------------------------------------------------------===//

def ByteIRTotalBufferizePipeline : Pass<"byteir-total-bufferize", "mlir::ModuleOp"> {
  let summary = "Performs all bufferization, including mhlo to lmhlo";
  let constructor = "mlir::createByteIRTotalBufferizePipelinePass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::lmhlo::LmhloDialect",
    "mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
//
// Pipeline passes of final passes (TBD)
//
//===----------------------------------------------------------------------===//


#endif // BYTEIR_TRANSFORMS_PASSES