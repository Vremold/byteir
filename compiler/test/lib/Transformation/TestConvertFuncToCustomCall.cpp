//===- TestConvertFuncToCustomCall.cpp ------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct FuncToCustomCallConverterTest : public FuncToCustomCallConverterLookup {

  FuncToCustomCallConverterTest(const llvm::StringMap<std::string> &externalMap)
      : FuncToCustomCallConverterLookup() {
    for (const auto &it : externalMap) {
      funcNameToCustomMeta.try_emplace(it.first(), it.second, false);
    }
  }
  virtual ~FuncToCustomCallConverterTest() {}
};

struct TestConvertFuncToCustomCallPass
    : public PassWrapper<TestConvertFuncToCustomCallPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertFuncToCustomCallPass)

  StringRef getArgument() const final { return "test-convert-func-to-custom"; }

  StringRef getDescription() const final {
    return "Convert Func to CustomCall";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    llvm::StringMap<std::string> funcNameToCallTarget;
    funcNameToCallTarget.try_emplace("test.test_name", "TestName");

    std::unique_ptr<FuncToCustomCallConverterBase> converter =
        std::make_unique<FuncToCustomCallConverterTest>(funcNameToCallTarget);

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertFuncToCustomCallPass(converter.get()));
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestConvertFuncToCustomCallPass() {
  PassRegistration<TestConvertFuncToCustomCallPass>();
}
} // namespace test
} // namespace byteir
