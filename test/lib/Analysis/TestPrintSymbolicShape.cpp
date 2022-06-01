//===- TestPrintSymbolicShape.cpp -----------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/SymbolicShape.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct TestPrintSymbolicShapePass
    : public PassWrapper<TestPrintSymbolicShapePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-print-symbolic-shape"; }

  StringRef getDescription() const final {
    return "Print the symbolic shape auxiliary functions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    SymbolicShapeAnalysis(op).dump(llvm::outs());
  }
};

} // end anonymous namespace

namespace byteir {
namespace test {
void registerTestPrintSymbolicShapePass() {
  PassRegistration<TestPrintSymbolicShapePass>();
}
} // namespace test
} // namespace byteir