//===- TestMhloCanonicalizeExt.cpp ----------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct TestMhloCanonicalizeExtPass
    : public PassWrapper<TestMhloCanonicalizeExtPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMhloCanonicalizeExtPass)

  StringRef getArgument() const final { return "test-mhlo-canonicalize-ext"; }

  StringRef getDescription() const final { return "Mhlo Canonicalize Ext"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
    registry.insert<mlir::shape::ShapeDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    mhlo::populateCanonicalizeExtPatterns(patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestMhloCanonicalizeExtPass() {
  PassRegistration<TestMhloCanonicalizeExtPass>();
}
} // namespace test
} // namespace byteir
