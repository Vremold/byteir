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
// ConvBiasActFusion
//===----------------------------------------------------------------------===//

def ConvBiasActFusion : Pass<"fuse-conv-bias-act", "mlir::FuncOp"> {
  let summary = "Fuse convolution bias activation";
  let constructor = "mlir::createConvBiasActFusionPass()";
}

//===----------------------------------------------------------------------===//
// DotTransposeFusion : this pass should run after HloTransposeDotToDotGeneral
//===----------------------------------------------------------------------===//

def DotTransposeFusion : Pass<"fuse-dot-transpose", "mlir::FuncOp"> {
  let summary = "Fuse dot/dot_general + transpose";
  let constructor = "mlir::createDotTransposeFusionPass()";
}

//===----------------------------------------------------------------------===//
// ElementFusion
//===----------------------------------------------------------------------===//

def ElementFusion : Pass<"fuse-element", "mlir::FuncOp"> {
  let summary = "Fuse elementwise op";
  let constructor = "mlir::createElementFusionPass()";
  let options = [
    Option<"attachTag", "attach-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag to be attached with the fused op">
  ];
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
// HloFolder
//===----------------------------------------------------------------------===//

// TODO(liuyuanqiang): re-implement this as folder
def HloFolder : Pass<"hlo-fold", "mlir::FuncOp"> {
  let summary = "Fold mhlo ops";
  let constructor = "mlir::createHloFolderPass()";
}


//===----------------------------------------------------------------------===//
// HloTransposeDotToDotGeneral
//===----------------------------------------------------------------------===//

def HloTransposeDotToDotGeneral : Pass<"hlo-transpose-dot-to-dot-general", "mlir::FuncOp"> {
  let summary = "Fuse transpose dot to dot_general";
  let constructor = "mlir::createHloTransposeDotToDotGeneralPass()";
}

//===----------------------------------------------------------------------===//
// ReduceFusion
//===----------------------------------------------------------------------===//

def ReduceFusion : Pass<"fuse-reduce", "mlir::FuncOp"> {
  let summary = "Fuse reduce and reduce-window ops";
  let constructor = "mlir::createReduceFusionPass()";
  let options = [
    Option<"attachTag", "attach-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag to be attached with the fused op">
  ];
}

#endif // BYTEIR_MHLO_PASSES