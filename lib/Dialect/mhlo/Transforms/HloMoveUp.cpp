//===- HloMoveUp.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalExt.h"
#include "byteir/Dialect/mhlo/Transforms/HloMove.h"
#include "byteir/Dialect/mhlo/Transforms/MoveCommon.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

// For now, we support single result, Elementwise,
// SameOperandsAndResultShape (avoid implicit broadcast)
inline bool isElementwiseOneResult(Operation *op) {
  return op->hasTrait<::mlir::OpTrait::Elementwise>() &&
         op->hasTrait<::mlir::OpTrait::SameOperandsAndResultShape>() &&
         op->hasTrait<::mlir::OpTrait::OneResult>();
}

struct TransposeMoveUpPattern : public HloMoveUpPattern<mhlo::TransposeOp> {
  TransposeMoveUpPattern(MLIRContext *context,
                         const llvm::DenseSet<llvm::StringRef> &blocker,
                         bool multiInput)
      : HloMoveUpPattern<mhlo::TransposeOp>(context, blocker, multiInput) {}

  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType(); // T2 as Transpose: T1 -> T2
    auto defOp = op.getOperand().getDefiningOp();

    // early termination
    // 1) op.getOperand() is an argument
    // 2) op.getOperand() has another user
    // 3) defOp is in the blockers
    if (defOp == nullptr || UseCount(op.getOperand()) > 1 ||
        blockers.contains(defOp->getName().getStringRef())) {
      return failure();
    }

    // See Line 28 comment
    if (!isElementwiseOneResult(defOp))
      return failure();

    // isElementwiseOneResult(defOp) == true
    SmallDenseSet<Value> constInputs;
    SmallDenseSet<Value> nonConstInputs;
    for (auto operand : defOp->getOperands()) {
      if (IsSplatMhloConstantValue(operand)) {
        if (!constInputs.contains(operand)) {
          constInputs.insert(operand);
        }
      } else {
        if (!nonConstInputs.contains(operand)) {
          nonConstInputs.insert(operand);
        }
      }
    }

    // terminate if assumes single input but has multiple
    if (!multiInput && nonConstInputs.size() > 1) {
      return failure();
    }

    BlockAndValueMapping bvm;
    // create all const and put into bvm
    for (auto input : constInputs) {
      ElementsAttr oldConstAttr = input.getDefiningOp<mhlo::ConstOp>().value();
      auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, resultType);
      auto newConstOp =
          rewriter.create<mhlo::ConstOp>(op->getLoc(), newConstAttr.getValue());
      bvm.map(input, newConstOp.output());
    }

    // clone new Transpose for nonConstInputs
    for (auto input : nonConstInputs) {
      BlockAndValueMapping bvmTrans;
      bvmTrans.map(op.getOperand(), input);
      auto newTransType =
          mixType(/*cloneFromElementType*/ input.getType().cast<ShapedType>(),
                  /*cloneFromShapes*/ op.getType());
      auto newTrans =
          cloneAndReplaceResultTypes(rewriter, op, bvmTrans, {newTransType});
      bvm.map(input, newTrans->getResult(0));
    }

    // clone a new elementwise as consumer
    auto maybeResultTypes =
        mixTypes(/*cloneFromElementTypes*/ defOp->getResultTypes(),
                 /*cloneFromShapes*/ op->getResultTypes());
    // maybeResultTypes should always have value
    assert(maybeResultTypes.hasValue());

    auto newConsumer = cloneAndReplaceResultTypes(rewriter, defOp, bvm,
                                                  maybeResultTypes.getValue());
    rewriter.replaceOp(op, newConsumer->getResults());
    return success();
  }
};

struct ReshapeMoveUpPattern : public HloMoveUpPattern<mhlo::ReshapeOp> {
  ReshapeMoveUpPattern(MLIRContext *context,
                       const llvm::DenseSet<llvm::StringRef> &blocker,
                       bool multiInput)
      : HloMoveUpPattern<mhlo::ReshapeOp>(context, blocker, multiInput) {}

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType(); // T2 as Reshape: T1 -> T2
    auto defOp = op.getOperand().getDefiningOp();

    // early termination
    // 1) op.getOperand() is an argument
    // 2) op.getOperand() has another user
    // 3) defOp is in the blockers
    if (defOp == nullptr || UseCount(op.getOperand()) > 1 ||
        blockers.contains(defOp->getName().getStringRef())) {
      return failure();
    }

    // See Line 28 comment
    if (!isElementwiseOneResult(defOp))
      return failure();

    // isElementwiseOneResult(defOp) == true
    SmallDenseSet<Value> constInputs;
    SmallDenseSet<Value> nonConstInputs;
    for (auto operand : defOp->getOperands()) {
      if (IsSplatMhloConstantValue(operand)) {
        if (!constInputs.contains(operand)) {
          constInputs.insert(operand);
        }
      } else {
        if (!nonConstInputs.contains(operand)) {
          nonConstInputs.insert(operand);
        }
      }
    }

    // terminate if assumes single input but has multiple
    if (!multiInput && nonConstInputs.size() > 1) {
      return failure();
    }

    BlockAndValueMapping bvm;
    // create all const and put into bvm
    for (auto input : constInputs) {
      ElementsAttr oldConstAttr = input.getDefiningOp<mhlo::ConstOp>().value();
      auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, resultType);
      auto newConstOp =
          rewriter.create<mhlo::ConstOp>(op->getLoc(), newConstAttr.getValue());
      bvm.map(input, newConstOp.output());
    }

    // clone new Reshape for nonConstInputs
    for (auto input : nonConstInputs) {
      BlockAndValueMapping bvmReshape;
      bvmReshape.map(op.getOperand(), input);

      auto newReshapeType =
          mixType(/*cloneFromElementType*/ input.getType().cast<ShapedType>(),
                  /*cloneFromShapes*/ op.getType());

      auto newReshape = cloneAndReplaceResultTypes(rewriter, op, bvmReshape,
                                                   {newReshapeType});
      bvm.map(input, newReshape->getResult(0));
    }

    // clone a new elementwise as consumer
    auto maybeResultTypes =
        mixTypes(/*cloneFromElementTypes*/ defOp->getResultTypes(),
                 /*cloneFromShapes*/ op->getResultTypes());
    // maybeResultTypes should always have value
    assert(maybeResultTypes.hasValue());

    auto newConsumer = cloneAndReplaceResultTypes(rewriter, defOp, bvm,
                                                  maybeResultTypes.getValue());
    rewriter.replaceOp(op, newConsumer->getResults());

    return success();
  }
};

struct HloMoveUpPass : public HloMoveUpBase<HloMoveUpPass> {

  HloMoveUpPass(bool supportMultiInput) : HloMoveUpBase() {
    multiInput = supportMultiInput;
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());

    // add pattern
    populateHloMoveUpPattern(patterns, {}, multiInput);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, funcOp.getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "HloMoveUpPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateHloMoveUpPattern(RewritePatternSet &patterns,
                                    const llvm::DenseSet<StringRef> &blocker,
                                    bool multiInput) {
  patterns.add<TransposeMoveUpPattern, ReshapeMoveUpPattern>(
      patterns.getContext(), blocker, multiInput);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHloMoveUpPass(bool multiInput) {
  return std::make_unique<HloMoveUpPass>(multiInput);
}
