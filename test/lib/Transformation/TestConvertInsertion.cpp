//===- TestConvertInsertion.cpp ------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvertInsertion.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

constexpr StringRef getByteIRUnitTestAttrName() {
  return "__byteir_unit_test__";
}

std::unique_ptr<ConvertRuleBase> unitTestRuleExample(MLIRContext *ctx) {
  auto collector = std::make_unique<ConvertOnlyCheckElementType>(
      getByteIRUnitTestAttrName());
  collector->convertElementType.try_emplace(Float32Type::get(ctx),
                                            Float16Type::get(ctx));
  return collector;
}
} // namespace

struct TestConvertInsertionPass
    : public PassWrapper<TestConvertInsertionPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const final { return "test-insert-convert"; }

  StringRef getDescription() const final {
    return "Test Insert ConvertOps (for CallOps and FuncOps now)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    auto ctx = m.getContext();
    std::unique_ptr<ConvertRuleBase> testRule = unitTestRuleExample(ctx);

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertInsertionPass(testRule.get()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

namespace byteir {
namespace test {
void registerTestConvertInsertionPass() {
  PassRegistration<TestConvertInsertionPass>();
}
} // namespace test
} // namespace byteir