//===- TestByreOpInterface.cpp --------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

struct TestByreOpInterfacePass
    : public PassWrapper<TestByreOpInterfacePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestByreOpInterfacePass)

  StringRef getArgument() const final { return "test-byre-op-interface"; }

  StringRef getDescription() const final { return "Test byre op interface"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<byre::ByreDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    op.walk([&](byre::ByreOp op) {
      llvm::outs() << op.getCalleeName() << '\n';
      auto inputs = op.getInputs();
      llvm::outs() << inputs.size() << ' ' << "Inputs:\n";
      for (auto &&input : inputs) {
        llvm::outs() << '\t' << input << '\n';
      }
      auto outputs = op.getOutputs();
      llvm::outs() << outputs.size() << ' ' << "Outputs:\n";
      for (auto &&output : outputs) {
        llvm::outs() << '\t' << output << '\n';
      }
    });
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestByreOpInterfacePass() {
  PassRegistration<TestByreOpInterfacePass>();
}
} // namespace test
} // namespace byteir
