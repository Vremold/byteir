//===- StaticShapeInference.cpp -------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/StaticShapeInference.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct StaticShapeInferencePass
    : public StaticShapeInferenceBase<StaticShapeInferencePass> {

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    (void)runShapeInference(funcOp, /*isStaticShapeInfer=*/false);
  };
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createStaticShapeInferencePass() {
  return std::make_unique<StaticShapeInferencePass>();
}
