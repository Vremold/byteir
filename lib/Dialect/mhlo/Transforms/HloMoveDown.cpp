//===- HloMoveDown.cpp ----------------------------------------*--- C++ -*-===//
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
#include <iostream>

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

struct TransposeMoveDownPattern : public HloMoveDownPattern<mhlo::TransposeOp> {
  TransposeMoveDownPattern(MLIRContext *context,
                           const llvm::DenseSet<llvm::StringRef> &blocker,
                           bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::TransposeOp>(context, blocker, allMultiUser,
                                              multiUser) {}

  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType = op.getOperand().getType(); // T1 as Transpose: T1 -> T2

    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && UserCount(value) != 1) {
      return failure();
    }

    SmallDenseSet<Operation *> users;
    for (auto user : value.getUsers()) {
      // skip checked user
      if (users.contains(user))
        continue;

      // skip if a user is
      // 1) a terminator or
      // 2) in the blockers
      if (user->hasTrait<::mlir::OpTrait::IsTerminator>() ||
          blockers.contains(user->getName().getStringRef())) {
        // if requiring allMultiUser legal, one skip implies failure
        if (allMultiUser)
          return failure();
        continue;
      }

      // just check ElementwiseOneResult
      // See Line 29 comment
      if (!isElementwiseOneResult(user)) {
        if (allMultiUser)
          return failure();
        continue;
      }

      // isElementwiseOneResult(user) == true
      bool failed = false;
      for (auto operand : user->getOperands()) {
        if (operand != value && !IsSplatMhloConstantValue(operand)) {
          if (allMultiUser)
            return failure();
          failed = true;
          break;
        }
      }
      if (failed)
        continue;
      users.insert(user);
    }

    // terminate if no legal users
    if (users.size() == 0)
      return failure();

    // process user
    for (auto user : users) {
      BlockAndValueMapping bvm;
      SmallDenseSet<Value> constInputs;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(value)) {
            bvm.map(value, op.getOperand());
          }
        } else {
          // IsSplatMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        ElementsAttr oldConstAttr =
            input.getDefiningOp<mhlo::ConstOp>().value();
        auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, operandType);
        auto newConstOp = rewriter.create<mhlo::ConstOp>(
            op->getLoc(), newConstAttr.getValue());
        bvm.map(input, newConstOp.output());
      }

      // clone an elementwise op as producer
      auto newProducer = cloneAndReplaceResultTypes(rewriter, user, bvm,
                                                    op->getOperandTypes());

      // create transpose op
      auto trans = rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
          user, op.getType(), newProducer->getResult(0), op.permutation());
    }

    return success();
  }
};

struct ReshapeMoveDownPattern : public HloMoveDownPattern<mhlo::ReshapeOp> {
  ReshapeMoveDownPattern(MLIRContext *context,
                         const llvm::DenseSet<llvm::StringRef> &blocker,
                         bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::ReshapeOp>(context, blocker, allMultiUser,
                                            multiUser) {}

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType = op.getOperand().getType(); // T1 as Reshape: T1 -> T2

    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && UserCount(value) != 1) {
      return failure();
    }

    SmallDenseSet<Operation *> users;
    for (auto user : value.getUsers()) {
      // skip checked user
      if (users.contains(user))
        continue;

      // skip if a user is
      // 1) a terminator or
      // 2) in the blockers
      if (user->hasTrait<::mlir::OpTrait::IsTerminator>() ||
          blockers.contains(user->getName().getStringRef())) {
        // if requiring allMultiUser legal, one skip implies failure
        if (allMultiUser)
          return failure();
        continue;
      }

      // just check ElementwiseOneResult
      // See Line 29 comment
      if (!isElementwiseOneResult(user)) {
        if (allMultiUser)
          return failure();
        continue;
      }

      // isElementwiseOneResult(user) == true
      bool failed = false;
      for (auto operand : user->getOperands()) {
        if (operand != value && !IsSplatMhloConstantValue(operand)) {
          if (allMultiUser)
            return failure();
          failed = true;
          break;
        }
      }
      if (failed)
        continue;
      users.insert(user);
    }

    // terminate if no legal users
    if (users.size() == 0)
      return failure();

    // process user
    for (auto user : users) {
      BlockAndValueMapping bvm;
      SmallDenseSet<Value> constInputs;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(value)) {
            bvm.map(value, op.getOperand());
          }
        } else {
          // IsSplatMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        ElementsAttr oldConstAttr =
            input.getDefiningOp<mhlo::ConstOp>().value();
        auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, operandType);
        auto newConstOp = rewriter.create<mhlo::ConstOp>(
            op->getLoc(), newConstAttr.getValue());
        bvm.map(input, newConstOp.output());
      }

      // clone an elementwise op as producer
      auto newProducer = cloneAndReplaceResultTypes(rewriter, user, bvm,
                                                    op->getOperandTypes());

      // create reshape op
      auto reshape = rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
          user, op.getType(), newProducer->getResult(0));
    }

    return success();
  }
};

struct HloMoveDownPass : public HloMoveDownBase<HloMoveDownPass> {

  HloMoveDownPass(bool supportAllMultiUsers, bool supportMultiUsers)
      : HloMoveDownBase() {
    allMultiUser = supportAllMultiUsers;
    multiUser = supportMultiUsers;
  }

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    OwningRewritePatternList patterns(funcOp.getContext());

    // add pattern
    populateHloMoveDownPattern(patterns, {}, allMultiUser, multiUser);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, patterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "HloMoveDownPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateHloMoveDownPattern(RewritePatternSet &patterns,
                                      const llvm::DenseSet<StringRef> &blocker,
                                      bool allMultiUser, bool multiUser) {
  patterns.add<TransposeMoveDownPattern, ReshapeMoveDownPattern>(
      patterns.getContext(), blocker, allMultiUser, multiUser);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHloMoveDownPass(bool allMultiUser, bool multiUser) {
  return std::make_unique<HloMoveDownPass>(allMultiUser, multiUser);
}
