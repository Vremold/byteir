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

def CoalescedForToGPULaunch : FunctionPass<"coalesced-for-to-gpu"> {
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
  ];
}

//===----------------------------------------------------------------------===//
// HloToLinalg
//===----------------------------------------------------------------------===//

def HloFusionToLinalg : FunctionPass<"hlo-fusion-to-linalg"> {
  let summary = "Convert from HLO fusion to Linalg dialect.";
  let constructor = "mlir::createHloFusionToLinalgPass()";
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag be applied the fused ops">,
  ];
}

def UnrealizedCastToLinalg : FunctionPass<"unrealized-cast-to-linalg"> {
  let summary = "Convert UnrealizedCast op generated from hlo lowering to Linalg";
  let constructor = "mlir::createUnrealizedCastToLinalgPass()";
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

#endif // BYTEIR_CONVERSION_PASSES