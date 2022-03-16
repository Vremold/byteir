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
// ClusterConstraint
//===----------------------------------------------------------------------===//

def ClusterConstraint : Pass<"cluster-constraint", "mlir::FuncOp"> {
  let summary = "Cluster mhlo ops with constraint";
  let description = [{
    This pass do some early check in the tensor world(i.e. mhlo) and also treat
    those constraint values as constants and fuse them into mhlo.fusion
  }];
  let constructor = "mlir::createClusterConstraintPass()";
}

//===----------------------------------------------------------------------===//
// ConvBackwardFusion
//===----------------------------------------------------------------------===//

def ConvBackwardFusion : Pass<"fuse-conv-backward", "mlir::FuncOp"> {
  let summary = "Fuse convolution backward data & filter";
  let constructor = "mlir::createConvBackwardFusionPass()";
}

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
    Option<"clusterSingleElemwiseOp", "cluster-single-elemwise-op", 
           "bool", /*default=*/"false",
           "whether to cluster single element-wise operation into mhlo.fusion op">,
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
// IOConvertFusion
//===----------------------------------------------------------------------===//

def IOConvertFusion : Pass<"fuse-io-convert", "mlir::FuncOp"> {
  let summary = "Fuse op with Input/Output convert";
  let constructor = "mlir::createIOConvertFusionPass()";
  let options = [
    Option<"opName", "op-name", "std::string", /*default=*/"",
           "Fuse op with Input/Output convert into mhlo.fusion">,
    ListOption<"inputArgIdx", "input-arg-idx", "int",
               "input args with convert: input-arg-index=0,1,...",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"outputArgIdx", "output-arg-idx", "int",
               "output args with convert: output-arg-index=0,1,...",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"byreComputeName", "byre-compute-name", "std::string", /*default=*/"",
            "Set mhlo.fusion with byre-compute-name">,
  ];
}

//===----------------------------------------------------------------------===//
// HloMoveDown
//===----------------------------------------------------------------------===//

def HloMoveDown : Pass<"hlo-move-down", "mlir::FuncOp"> {
  let summary = "Move selected mhlo op down (to output)";
  let constructor = "mlir::createHloMoveDownPass()";
  let options = [
    Option<"allMultiUser", "all-multi-user", 
           "bool", /*default=*/"false",
           "whether to support multiple users and require all user legal">,
    Option<"multiUser", "multi-user", 
           "bool", /*default=*/"false",
           "whether to support multiple users, it might be overridden by all-multi-user">,
  ];
}

//===----------------------------------------------------------------------===//
// HloMoveUp
//===----------------------------------------------------------------------===//

def HloMoveUp : Pass<"hlo-move-up", "mlir::FuncOp"> {
  let summary = "Move selected mhlo op up (to input)";
  let constructor = "mlir::createHloMoveUpPass()";
  let options = [
    Option<"multiInput", "multi-input", 
           "bool", /*default=*/"false",
           "whether to support multiple inputs">,
  ];
}

//===----------------------------------------------------------------------===//
// ReduceFusion
//===----------------------------------------------------------------------===//

def ReduceFusion : Pass<"fuse-reduce", "mlir::FuncOp"> {
  let summary = "Fuse reduce and reduce-window ops";
  let constructor = "mlir::createReduceFusionPass()";
}

//===----------------------------------------------------------------------===//
// RewriteWithConstraint
//===----------------------------------------------------------------------===//

def RewriteWithConstraint :  Pass<"rewrite-with-constraint", "mlir::FuncOp"> {
  let summary = "Rewrite operation with constraint";
  let constructor = "mlir::createRewriteWithConstraintPass()";
}

//===----------------------------------------------------------------------===//
// TrivialFusion
//===----------------------------------------------------------------------===//

def TrivialFusion : Pass<"fuse-trivial", "mlir::FuncOp"> {
  let summary = "Fuse trivial single ops";
  let constructor = "mlir::createTrivialFusionPass()";
}

#endif // BYTEIR_MHLO_PASSES