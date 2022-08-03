//===- TrivialFusion.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/TrivialFusion.h"
#include "PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

template <typename OpTy>
struct SingleOpPattern : public OpRewritePattern<OpTy> {
  SingleOpPattern(MLIRContext *context,
                  const llvm::DenseMap<StringRef, StringRef> &lut)
      : OpRewritePattern<OpTy>(context), src_to_name_(lut) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    // avoid already fused
    if (op->template getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    MhloFusionPattern pattern;
    pattern.push_back(op);
    auto fusion = createMhloFusionFromPattern(rewriter, pattern);
    // add attr
    fusion->setAttr(getByteIRTrivialFusionAttrName(),
                    UnitAttr::get(fusion.getContext()));

    // FIXME make this optional base on OpTy
    fusion->setAttr(byre::getByreForceComputeNameAttrName(),
                    UnitAttr::get(fusion.getContext()));

    auto found = src_to_name_.find(op.getOperation()->getName().getStringRef());
    if (found != src_to_name_.end()) {
      fusion->setAttr(byre::getByreComputeName(),
                      rewriter.getStringAttr(found->second));
    }

    return success();
  }

  const llvm::DenseMap<StringRef, StringRef> &src_to_name_;
};

struct TrivialFusionPass : public TrivialFusionBase<TrivialFusionPass> {

  TrivialFusionPass() : TrivialFusionBase() {
    // TODO: change to loading from outside
    // mhloNameMap.insert({"mhlo.rng_bit_generator", "RngBitGeneratorOp"});
    // mhloNameMap.insert({"mhlo.rng_normal", "RngNormalOp"});
    // mhloNameMap.insert({"mhlo.rng_uniform", "RngUniform"});
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateTrivialFusionPattern(patterns, mhloNameMap);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "TrivialFusionPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }

  llvm::DenseMap<StringRef, StringRef> mhloNameMap;
};
} // namespace

void mlir::populateTrivialFusionPattern(
    RewritePatternSet &patterns, llvm::DenseMap<StringRef, StringRef> &lut) {
  // FIXME: mhlo::RngNormalOp and RngUninformOp has been merged into RngOp.
  // patterns.add<SingleOpPattern<mhlo::RngBitGeneratorOp>,
  //              SingleOpPattern<mhlo::RngNormalOp>,
  //              SingleOpPattern<mhlo::RngUniformOp>>(patterns.getContext(),
  //              lut);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createTrivialFusionPass() {
  return std::make_unique<TrivialFusionPass>();
}
