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
    "mlir::scf::SCFDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Byre Host Pipeline
//===----------------------------------------------------------------------===//

def ByreHostPipeline : Pass<"byre-host", "mlir::ModuleOp"> {
  let summary = "Byre Host Pipeline";
  let constructor = "mlir::createByreHostPipelinePass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "An optional string to speicify entry function.">,
    Option<"deviceFile", "device-file-name", "std::string",
            /*default=*/"\"kernel\"",
            "An optional string to speicify device file name.">,
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to target device.">,
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
    "mlir::lace::LaceDialect",
    "mlir::memref::MemRefDialect",
  ];
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "An optional string to speicify entry function.">,
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,
    Option<"disableMemoryPlanning", "disable-memory-planning", "bool", /*default=*/"false",
           "disable memory planning">,
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
// Linalg Tensor Opt Pipeline
//===----------------------------------------------------------------------===//

def LinalgTensorOptPipeline : Pass<"linalg-tensor-opt", "mlir::ModuleOp"> {
  let summary = "Linalg Opt Pipeline in Tensor";
  let constructor = "mlir::createLinalgTensorOptPipelinePass()";
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
// SCF Opt Pipeline
//===----------------------------------------------------------------------===//

def SCFOptPipeline : Pass<"scf-opt", "mlir::ModuleOp"> {
  let summary = "SCF Opt Pipeline";
  let constructor = "mlir::createSCFOptPipelinePass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
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

//===----------------------------------------------------------------------===//
// Byteir All Opt Pipeline. 
//===----------------------------------------------------------------------===//

def ByteirAllOptPipeline : Pass<"byteir-all-opt", "mlir::ModuleOp"> {
  let summary = "Byteir all Opt Pipeline";
  let constructor = "mlir::createByteIRAllOptPipelinePass()";
  let dependentDialects = [
    "mlir::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::byre::ByreDialect",
    "mlir::gpu::GPUDialect",
    "mlir::lace::LaceDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::lmhlo::LmhloDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "An optional string to speicify entry function.">,
    Option<"target", "target", "std::string",
            /*default=*/"",
            "An optional attribute to speicify target.">,
  ];
}



#endif // BYTEIR_PIPELINES_PASSES