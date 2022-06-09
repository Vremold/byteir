//===- TestPrintSideEffect.cpp --------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Analysis/SideEffect.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct TestPrintArgSideEffectPass
    : public PassWrapper<TestPrintArgSideEffectPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "test-print-arg-side-effect"; }

  StringRef getDescription() const final { return "Print the arg side effect"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::lmhlo::LmhloDialect>();
  }

  void runOnOperation() override {
    auto &os = llvm::outs();
    ModuleOp m = getOperation();
    ArgSideEffectAnalysis analysis;
    analysis.dump(llvm::outs());
    os << "============= Test Module"
       << " =============\n";
    for (auto f : m.getOps<FuncOp>()) {
      for (auto &block : f.getBlocks()) {
        for (auto &op : block.without_terminator()) {
          os << "Testing " << op.getName() << ":\n";
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            auto argSETy = analysis.getType(&op, i);
            os << "arg " << i << " ArgSideEffectType: " << to_str(argSETy)
               << "\n";
          }
        }
      }
    }
  }
};

} // end anonymous namespace

namespace byteir {
namespace test {
void registerTestPrintArgSideEffectPass() {
  PassRegistration<TestPrintArgSideEffectPass>();
}
} // namespace test
} // namespace byteir