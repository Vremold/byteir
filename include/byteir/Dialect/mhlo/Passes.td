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
// Bounded Shape Inference
//===----------------------------------------------------------------------===//

def BoundedShapeInference : Pass<"bounded-shape-infer", "FuncOp"> {
  let summary = "Bounded shape inference";
  let constructor = "mlir::createBoundedShapeInferencePass()";
}

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
// ConvertInsertion (for CallOps and FunOps now)
//===----------------------------------------------------------------------===//

//def ConvertInsertion : Pass<"insert-convert", "ModuleOp"> {
//  let summary = "Insert ConvertOps (for CallOps and FuncOps now)";
//  let constructor = "mlir::createConvertInsertionPass()";
//  let dependentDialects = [
//    "mlir::mhlo::MhloDialect"
//  ];
//}

//===----------------------------------------------------------------------===//
// DotTransposeFusion : this pass should run after HloTransposeDotToDotGeneral
//===----------------------------------------------------------------------===//

def DotTransposeFusion : Pass<"fuse-dot-transpose", "mlir::FuncOp"> {
  let summary = "Fuse dot/dot_general + transpose";
  let constructor = "mlir::createDotTransposeFusionPass()";
}

//===----------------------------------------------------------------------===//
// FusionOutlining
//===----------------------------------------------------------------------===//

def FusionOutlining : Pass<"fusion-outlining", "ModuleOp"> {
  let summary = "Outline mhlo FusionOp to a FuncOp";
  let constructor = "mlir::createFusionOutliningPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
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
// LayoutTransformation
//===----------------------------------------------------------------------===//

def LayoutTransformation : Pass<"transform-layout", "mlir::FuncOp"> {
  let summary = "Layout Transformation: Conv2d, BatchNormTraining";
  let constructor = "mlir::createLayoutTransformationPass()";
  let options = [
    Option<"targetLayout", "target-layout", "std::string", /*default=*/"",
           "Target Layout">,
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
// ShapeReification
//===----------------------------------------------------------------------===//

def ShapeReification : Pass<"shape-reification", "FuncOp"> {
  let summary = "Iteratively reify all shape computations.";
  let constructor = "mlir::createShapeReificationPass()";
  let dependentDialects = [
    "mlir::shape::ShapeDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Static Shape Inference
//===----------------------------------------------------------------------===//

def StaticShapeInference : Pass<"static-shape-infer", "FuncOp"> {
  let summary = "Static shape inference";
  let constructor = "mlir::createStaticShapeInferencePass()";
}

//===----------------------------------------------------------------------===//
// TrivialFusion
//===----------------------------------------------------------------------===//

def TrivialFusion : Pass<"fuse-trivial", "mlir::FuncOp"> {
  let summary = "Fuse trivial single ops";
  let constructor = "mlir::createTrivialFusionPass()";
}

#endif // BYTEIR_MHLO_PASSES