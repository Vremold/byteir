//===- SymbolicShape.h ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_ANALYSIS_SYMBOLICSHAPE_H
#define BYTEIR_ANALYSIS_SYMBOLICSHAPE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

// [Deperated] Use ShapeOpt pipeline as a new version for the dynamic shape
// optimization.
// A auxiliary symbolic shape inference function will be created
// for each original function, and a ShapeReification pass will be run on the
// newly created function. A table mapping the value in the original function to
// the corresponding symbolic shape will also be created for later query and
// analysis.
// Ex. Let's say the original function is as below:
// clang-format off
// func @simple(%arg0: tensor<?x2xf32>, %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
//   %0 = mhlo.add %arg0, %arg1 : tensor<?x2xf32>
//   return %0 : tensor<?x12xf32>
// }
// Then the created auxiliary shape infer will be
// func private @_shape_infer_simple(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> (!shape.shape, tensor<?x4xf32>)  {
//   %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
//   %1 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
//   return %2, %0 : !shape.shape, tensor<?x4xf32>
// }
// clang-format on
class SymbolicShapeAnalysis {
public:
  explicit SymbolicShapeAnalysis(ModuleOp moduleOp);

  /// Delete all auxiliary function in destructor
  virtual ~SymbolicShapeAnalysis();

  /// Get the symbolic value in auxiliary shape infer function for v in an
  /// original function.
  Value getSymbolicShape(Value v);

  /// Construct and get a symbolic sources set for every intermediate values in
  /// the original functions.
  DenseMap<Value, DenseSet<Value>> constructSymbolicExprSourcesTable();

  /// Dumps the symbolic shape information to the given stream.
  void dump(raw_ostream &os);

private:
  void constructSymbolicShapeTable(FuncOp originalFunc,
                                   FuncOp symbolicShapeInferFunc);
  DenseSet<Value> findSymbolicExprSourcesRecursively(
      Value symbolicShape,
      DenseMap<Value, DenseSet<Value>> &symbolicShapeFnCache,
      const DenseMap<Value, Value> &auxiValToOrigin);

  ModuleOp moduleOp_;
  DenseMap<FuncOp, FuncOp> originalFuncToAuxiliary_;

  // A table mapping value in original function to symbolic shape in
  // corresponding auxiliary shape infer function. The symbolic shape is stored
  // as an pointer to an OpOperand of the terminator in case other intermediate
  // ops be modified after some passes.
  DenseMap<Value, OpOperand *> symbolisShapeTable_;
};

} // namespace mlir

#endif // BYTEIR_ANALYSIS_SYMBOLICSHAPE_H