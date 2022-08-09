//===- TestPrintShapeAnalysis.cpp -----------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/ShapeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
struct TestPrintShapeAnalysisPass
    : public PassWrapper<TestPrintShapeAnalysisPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-print-shape-analysis"; }

  StringRef getDescription() const final { return "Print the shape analysis."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() override {
    Operation *top = getOperation();

    DataFlowSolver solver;
    solver.load<ShapeAnalysis>();
    solver.load<ShapeValueAnalysis>();
    solver.load<DeadCodeAnalysis>();
    if (failed(solver.initializeAndRun(top)))
      return signalPassFailure();
    top->walk([&](Operation *op) {
      if (llvm::isa<InferShapedTypeOpInterface>(op)) {
        llvm::outs() << "for operation : " << *op
                     << ", inferred shapes are:\n\t";
        for (Value value : op->getResults()) {
          if (auto lattice = solver.lookupState<ShapeLattice>(value)) {
            if (!lattice->isUninitialized()) {
              lattice->getValue().print(llvm::outs());
            }
          }
        }
        llvm::outs() << "\n";
      }
      if (op->getNumResults()) {
        llvm::outs() << "for operation : " << *op
                     << ", inferred values are:\n\t";
        for (Value value : op->getResults()) {
          if (auto lattice =
                  solver.lookupState<Lattice<ConstantValue>>(value)) {
            if (!lattice->isUninitialized()) {
              lattice->getValue().print(llvm::outs());
            }
          }
        }
        llvm::outs() << "\n";
      }
    });
  }
};
} // namespace

namespace byteir {
namespace test {
void registerTestPrintShapeAnalysisPass() {
  PassRegistration<TestPrintShapeAnalysisPass>();
}
} // namespace test
} // namespace byteir
