//===- Passes.td - Conversion pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_PASSES
#define BYTEIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToGPU
//===----------------------------------------------------------------------===//

def CoalescedForToGPULaunch : Pass<"coalesced-for-to-gpu", "mlir::FuncOp"> {
  let summary = "Convert from a coalesced affine loop to gpu launch";
  let constructor = "mlir::createCoalescedForToGPULaunchPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
      Option<"blockSize", "bsize", "int64_t",  /*default=*/"32",
             "block size for GPU">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def GPUToNVVMExt : Pass<"gpu-to-nvvm-ext", "mlir::gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createGPUToNVVMExtPass()";
  let dependentDialects = ["NVVM::NVVMDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

//===----------------------------------------------------------------------===//
// HloToLHlo
//===----------------------------------------------------------------------===//

def ConvertHloToLHlo : Pass<"convert-hlo-to-lhlo", "ModuleOp"> {
  let summary = "An extension to convert hlo to lhlo Op.";
  let constructor = "mlir::createConvertHloToLHloPass()";
  let dependentDialects = [
    "mlir::lmhlo::LmhloDialect",
    "mlir::memref::MemRefDialect",
    "mlir::shape::ShapeDialect",
    "mlir::bufferization::BufferizationDialect",
  ];
}

//===----------------------------------------------------------------------===//
// HloToLinalg
//===----------------------------------------------------------------------===//

def HloFusionToLinalg : Pass<"hlo-fusion-to-linalg", "mlir::FuncOp"> {
  let summary = "Convert from HLO fusion to Linalg dialect.";
  let constructor = "mlir::createHloFusionToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag be applied the fused ops">,
  ];
}

def UnrealizedCastToLinalg : Pass<"unrealized-cast-to-linalg", "mlir::FuncOp"> {
  let summary = "Convert UnrealizedCast op generated from hlo lowering to Linalg";
  let constructor = "mlir::createUnrealizedCastToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToAce
//===----------------------------------------------------------------------===//

def ConvertUnregisteredToAce : Pass<"convert-unregistered-to-ace", "FuncOp"> {
  let summary = "Convert unregistered op to Ace Op.";
  let constructor = "mlir::createConvertUnregisteredToAcePass()";
  let dependentDialects = [
    "mlir::ace::AceDialect",
  ];
}

def ConvertMhloToAce : Pass<"convert-mhlo-to-ace", "FuncOp"> {
  let summary = "Fuse mhlo ops to coarse ace ops.";
  let constructor = "mlir::createConvertMhloToAcePass()";
  let dependentDialects = [
    "mlir::ace::AceDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToByre
//===----------------------------------------------------------------------===//

def ConvertToByre : Pass<"convert-to-byre", "ModuleOp"> {
  let summary = "Convert op to Byre Op.";
  let constructor = "mlir::createConvertToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
  ];
  let options = [
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,	
  ];

}

//===----------------------------------------------------------------------===//
// ToPTX or CUDA
//===----------------------------------------------------------------------===//

def GenPTXConfig : Pass<"gen-ptx-config", "mlir::FuncOp"> {
  let summary = "Generate PTX configuration";
  let constructor = "mlir::createGenPTXConfigPass()";
}

def CollectGPUKernel : Pass<"collect-gpu-kernel", "ModuleOp"> {
  let summary = "Collect GPU kernels for generating device code";
  let constructor = "mlir::createCollectGPUKernelPass()";
  let options = [
    Option<"moduleName", "module-name", "std::string",
            /*default=*/"\"unified\"",
            "Optional name for GPUModule to put all gpu kernels">,
  ];
}


#endif // BYTEIR_CONVERSION_PASSES