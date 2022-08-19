//===- TorchXLA.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall/TorchXLA.h"
#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

#include "../PassDetail.h"

// tentative
// FIXME: remove the entire file and folder after TorchXLA pipeline settled down

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct FuncToCustomCallConverterTorchXLA
    : public FuncToCustomCallConverterLookup {

  FuncToCustomCallConverterTorchXLA(
      const llvm::StringMap<std::string> &externalMap)
      : FuncToCustomCallConverterLookup() {
    for (const auto &it : externalMap) {
      funcNameToCustomMeta.try_emplace(it.first(), it.second, false);
    }
  }
  virtual ~FuncToCustomCallConverterTorchXLA() {}
};

struct ConvertFuncToCustomCallTorchXLAPass
    : public ConvertFuncToCustomCallTorchXLABase<
          ConvertFuncToCustomCallTorchXLAPass> {

  void runOnOperation() override {
    auto m = getOperation();
    llvm::StringMap<std::string> funcNameToCallTarget;
    funcNameToCallTarget.try_emplace("aten.erf", "byteir.erf");

    std::unique_ptr<FuncToCustomCallConverterBase> converter =
        std::make_unique<FuncToCustomCallConverterTorchXLA>(
            funcNameToCallTarget);

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertFuncToCustomCallPass(converter.get()));
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToCustomCallTorchXLAPass() {
  return std::make_unique<ConvertFuncToCustomCallTorchXLAPass>();
}
