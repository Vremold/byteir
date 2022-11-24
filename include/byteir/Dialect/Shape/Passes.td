//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_SHAPE_PASSES
#define BYTEIR_SHAPE_PASSES

include "mlir/Pass/PassBase.td"
include "byteir/Dialect/Shape/IR/ShapeExtBase.td"

//===----------------------------------------------------------------------===//
// ResolveShapeConstraint
//===----------------------------------------------------------------------===//

def ResolveShapeConstraint : Pass<"resolve-shape-constraint", "mlir::func::FuncOp"> {
  let summary = "Resolve shape constraint represented by MeetOp";
  let constructor = "mlir::createResolveShapeConstraintPass()";
}

//===----------------------------------------------------------------------===//
// InsertTieShape
//===----------------------------------------------------------------------===//

def InsertTieShape : Pass<"insert-tie-shape", "mlir::func::FuncOp"> {
  let summary = "Insert TieOp";
  let description = [{
    If a Value has dynamic shape, add a TieOp for it. The first operand of the
    TieOp is the Value, and the following operands are the symbolic expression
    of the dynamic dims. Currently the dynamic dims are only the results of
    tensor.dim, and they'll be reified later and optimized later.
  }];
  let constructor = "mlir::createInsertTieShapePass()";
  let dependentDialects = [
    "mlir::shape_ext::ShapeExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// SetAssumingAlwaysTrue
//===----------------------------------------------------------------------===//

def SetAssumingAlwaysTrue: Pass<"set-assuming-always-true", "mlir::func::FuncOp"> {
  let summary = "Make shape.assuming always true";
  let description = [{
    set the true to witness in shape.assuming
  }];
  let constructor = "mlir::createSetAssumingAlwaysTruePass()";
  let dependentDialects = [
    "mlir::shape_ext::ShapeExtDialect",
    "mlir::tensor::TensorDialect",
    "mlir::shape::ShapeDialect",
  ];
}

#endif // BYTEIR_SHAPE_PASSES
