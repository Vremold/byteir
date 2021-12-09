//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_MHLO_PASSES
#define BYTEIR_MHLO_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// HloFolder
//===----------------------------------------------------------------------===//

// TODO(liuyuanqiang): re-implement this as folder
def HloFolder : FunctionPass<"hlo-fold"> {
  let summary = "Fold mhlo ops";
  let constructor = "mlir::createHloFolderPass()";
}

//===----------------------------------------------------------------------===//
// FusionOutlining
//===----------------------------------------------------------------------===//

def FusionOutlining : Pass<"fusion-outlining", "ModuleOp"> {
  let summary = "Outline mhlo FusionOp to a FuncOp";
  let constructor = "mlir::createFusionOutliningPass()";
  let dependentDialects = [
    "StandardOpsDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ElementFusion
//===----------------------------------------------------------------------===//

def ElementFusion : FunctionPass<"fuse-element"> {
  let summary = "Fuse elementwise op";
  let constructor = "mlir::createElementFusionPass()";
  let options = [
    Option<"attachTag", "attach-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag to be attached with the fused op">
  ];
}

//===----------------------------------------------------------------------===//
// TransposeDotFusion
//===----------------------------------------------------------------------===//

def TransposeDotFusion : FunctionPass<"fuse-transpose-dot"> {
  let summary = "Fuse dot op with transpose";
  let constructor = "mlir::createTransposeDotFusionPass()";
}

//===----------------------------------------------------------------------===//
// ConvBiasActFusion
//===----------------------------------------------------------------------===//

def ConvBiasActFusion : FunctionPass<"fuse-conv-bias-act"> {
  let summary = "Fuse convolution bias activation";
  let constructor = "mlir::createConvBiasActFusionPass()";
}

#endif // BYTEIR_MHLO_PASSES