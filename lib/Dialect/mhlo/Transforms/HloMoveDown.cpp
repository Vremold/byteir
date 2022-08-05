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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <numeric>

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

static constexpr char kMoveDownDisableKey[] = "__move_down_disable__";

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
        if (operand != value && !isSplatMhloConstantValue(operand)) {
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
          // isSplatMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        ElementsAttr oldConstAttr =
            input.getDefiningOp<mhlo::ConstantOp>().value();
        auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, operandType);
        auto newConstOp = rewriter.create<mhlo::ConstantOp>(
            op->getLoc(), newConstAttr.getValue());
        bvm.map(input, newConstOp.output());
      }

      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.hasValue());

      // clone an elementwise op as producer
      auto newProducer = cloneAndReplaceResultTypes(
          rewriter, user, bvm, maybeResultTypes.getValue());

      // create transpose op
      auto trans = rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
          user, user->getResultTypes(), newProducer->getResult(0),
          op.permutation());
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
    if (op->hasAttr(kMoveDownDisableKey)) {
      return failure();
    }
    auto value = op.getResult();
    auto operandType = op.getOperand().getType(); // T1 as Reshape: T1 -> T2

    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && UserCount(value) != 1) {
      return failure();
    }

    SmallDenseSet<Operation *> users;
    SmallDenseSet<Value> reshapeInsertOperands;
    const auto isStaticArg = [](Value value) {
      if (!value || value.getDefiningOp() != nullptr) {
        return false;
      }
      const auto inputTy = value.getType().dyn_cast<RankedTensorType>();
      return inputTy && inputTy.hasStaticShape();
    };

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
        if (operand != value) {
          if (isSplatMhloConstantValue(operand)) {
            continue;
          }
          // fairly strict condition, so far we only accept static arg
          // to avoid side-effect on other branches as it seems we dont
          // know benefits besides branch here.
          // TODO(@zhangzhiwei.177): shall we remove static restriction?
          if (isStaticArg(operand)) {
            reshapeInsertOperands.insert(operand);
            continue;
          }
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
      SmallDenseSet<Value> insertedReshapeInputs;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(value)) {
            bvm.map(value, op.getOperand());
          }
        } else {
          // isSplatMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (reshapeInsertOperands.contains(operand) &&
              !insertedReshapeInputs.contains(operand)) {
            insertedReshapeInputs.insert(operand);
          } else if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        ElementsAttr oldConstAttr =
            input.getDefiningOp<mhlo::ConstantOp>().value();
        auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, operandType);
        auto newConstOp = rewriter.create<mhlo::ConstantOp>(
            op->getLoc(), newConstAttr.getValue());
        bvm.map(input, newConstOp.output());
      }
      for (auto input : insertedReshapeInputs) {
        const auto afterReshapeType = operandType.cast<RankedTensorType>();
        auto newReshapeOp = rewriter.create<mhlo::ReshapeOp>(
            rewriter.getUnknownLoc(), afterReshapeType, input);
        newReshapeOp->setAttr(kMoveDownDisableKey, rewriter.getBoolAttr(true));
        bvm.map(input, newReshapeOp.getResult());
      }

      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.hasValue());

      // clone an elementwise op as producer
      auto newProducer = cloneAndReplaceResultTypes(
          rewriter, user, bvm, maybeResultTypes.getValue());

      // create reshape op
      auto reshape = rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
          user, user->getResultTypes(), newProducer->getResult(0));
    }

    return success();
  }
};

struct BroadcastMoveDownPattern
    : public HloMoveDownPattern<mhlo::BroadcastInDimOp> {
  BroadcastMoveDownPattern(MLIRContext *context,
                           const llvm::DenseSet<llvm::StringRef> &blocker,
                           bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::BroadcastInDimOp>(context, blocker,
                                                   allMultiUser, multiUser) {}

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType =
        op.getOperand().getType(); // T1 as BroadcastInDim: T1 -> T2

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
        if (operand != value && !isSplatMhloConstantValue(operand)) {
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
          // isSplatMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        ElementsAttr oldConstAttr =
            input.getDefiningOp<mhlo::ConstantOp>().value();
        auto newConstAttr = cloneSplatElementsAttr(oldConstAttr, operandType);
        auto newConstOp = rewriter.create<mhlo::ConstantOp>(
            op->getLoc(), newConstAttr.getValue());
        bvm.map(input, newConstOp.output());
      }

      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.hasValue());

      // clone an elementwise op as producer
      auto newProducer = cloneAndReplaceResultTypes(
          rewriter, user, bvm, maybeResultTypes.getValue());

      // create broadcast op
      auto newOp = rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
          user, user->getResultTypes(), newProducer->getResult(0),
          op.broadcast_dimensions());
    }

    return success();
  }
};

struct BroadcastBinaryMoveDownPattern
    : public HloMoveDownPattern<mhlo::BroadcastInDimOp> {
  BroadcastBinaryMoveDownPattern(MLIRContext *context,
                                 const llvm::DenseSet<llvm::StringRef> &blocker,
                                 bool allMultiUser = false,
                                 bool multiUser = false)
      : HloMoveDownPattern<mhlo::BroadcastInDimOp>(context, blocker,
                                                   allMultiUser, multiUser) {}

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();

    // terminate if multi-users
    if (UserCount(value) != 1) {
      return failure();
    }

    auto consumer = *(value.user_begin());

    // terminate if not binary elementwise operator
    if (!(isElementwiseOneResult(consumer) &&
          consumer->getNumOperands() == 2)) {
      return failure();
    }

    ::mlir::Value lhsValue = consumer->getOperand(0);
    ::mlir::Value rhsValue = consumer->getOperand(1);

    // apply only if current op is the first operand
    if (value != lhsValue) {
      return failure();
    }

    // rhs also has to be a broadcast
    if (!rhsValue.getDefiningOp<BroadcastInDimOp>()) {
      return failure();
    }

    auto lhs = lhsValue.getDefiningOp<BroadcastInDimOp>();
    auto rhs = rhsValue.getDefiningOp<BroadcastInDimOp>();

    // lhs and rhs must have the same attribtue
    if (lhs.broadcast_dimensions() != rhs.broadcast_dimensions()) {
      return failure();
    }

    // all conditions are satisfied, rewrite
    BlockAndValueMapping bvm;
    bvm.map(lhs, lhs.getOperand());
    bvm.map(rhs, rhs.getOperand());

    rewriter.setInsertionPoint(consumer);
    auto newProducer = cloneAndReplaceResultTypes(rewriter, consumer, bvm,
                                                  op->getOperandTypes());

    auto newConsumer = rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        consumer, op.getType(), newProducer->getResult(0),
        op.broadcast_dimensions());

    return success();
  }
};

inline bool checkReshapeRemoveFirstNumberOneDimension(ReshapeOp op) {
  ArrayRef<int64_t> inShape =
      op.getOperand().getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> outShape =
      op.getResult().getType().cast<RankedTensorType>().getShape();
  bool is_remove_first =
      (outShape.size() == (inShape.size() - 1)) && (inShape[0] == 1);
  for (size_t i = 1; i < inShape.size(); ++i) {
    is_remove_first = (is_remove_first && (inShape[i] == outShape[i - 1]));
  }
  return is_remove_first;
}

/*
 * Before transform:
 *   broadcast -> reshape
 * After transform:
 *   reshape -> broadcast
 */
struct BroadcastReshapeMoveDownPattern
    : public HloMoveDownPattern<mhlo::BroadcastInDimOp> {
  BroadcastReshapeMoveDownPattern(
      MLIRContext *context, const llvm::DenseSet<llvm::StringRef> &blocker,
      bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::BroadcastInDimOp>(context, blocker,
                                                   allMultiUser, multiUser) {}

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType = op.getOperand().getType();

    // terminate if multi-users
    if (UserCount(value) != 1) {
      return failure();
    }

    auto consumer = *(value.user_begin());

    // consumer has to be a reshape
    mhlo::ReshapeOp reshape = dyn_cast<mhlo::ReshapeOp>(consumer);
    if (!reshape) {
      return failure();
    }

    // make sure broadcast do not touch the first dimension
    DenseIntElementsAttr bcastDim = op.broadcast_dimensions();
    if (*(bcastDim.begin()) != 0) {
      return failure();
    }

    // check the reshape just remove the first 1 dimension
    if (!checkReshapeRemoveFirstNumberOneDimension(reshape)) {
      return failure();
    }

    ArrayRef<int64_t> ishape = operandType.cast<RankedTensorType>().getShape();
    ArrayRef<int64_t> oshape_reshape =
        reshape.getType().cast<RankedTensorType>().getShape();

    // infer new output shape of reshape
    SmallVector<int64_t> newReshapeOShape;
    for (size_t i = 1; i < ishape.size(); ++i) {
      newReshapeOShape.push_back(ishape[i]);
    }
    RankedTensorType newReshapeOType = RankedTensorType::get(
        newReshapeOShape,
        operandType.cast<RankedTensorType>().getElementType());

    // infer the new broadcast dimensions
    SmallVector<int64_t> newBCastDim;
    for (auto it = bcastDim.begin() + 1; it < bcastDim.end(); ++it) {
      newBCastDim.push_back((*it).getSExtValue() - 1);
    }
    DenseIntElementsAttr new_bcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(newBCastDim.size())},
                              rewriter.getI64Type()),
        newBCastDim);

    // all conditions are satisfied, rewrite
    BlockAndValueMapping bvm;
    bvm.map(value, op.getOperand());

    auto newProducer =
        cloneAndReplaceResultTypes(rewriter, reshape, bvm, newReshapeOType);

    RankedTensorType new_otype_bcast = RankedTensorType::get(
        oshape_reshape, operandType.cast<RankedTensorType>().getElementType());

    auto newConsumer = rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        consumer, new_otype_bcast, newProducer->getResult(0), new_bcast_attr);

    return success();
  }
};

struct ReshapeBroadcastDotMoveDownPattern
    : public HloMoveDownPattern<mhlo::DotOp> {
  ReshapeBroadcastDotMoveDownPattern(
      MLIRContext *context, const llvm::DenseSet<llvm::StringRef> &blocker,
      bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::DotOp>(context, blocker, allMultiUser,
                                        multiUser) {}

  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    BroadcastInDimOp bcast = op.getOperand(0).getDefiningOp<BroadcastInDimOp>();
    if (!bcast) {
      return failure();
    }
    ReshapeOp reshape = bcast.getOperand().getDefiningOp<ReshapeOp>();
    if (!reshape) {
      return failure();
    }
    ::mlir::Value input = reshape.getOperand();
    ::mlir::Value weight = op.getOperand(1);
    Type dtype = input.getType().cast<RankedTensorType>().getElementType();

    if (!checkReshapeRemoveFirstNumberOneDimension(reshape)) {
      return failure();
    }
    if (!checkBroadcastFirstDimension(bcast)) {
      return failure();
    }

    // all conditions are satisfied, rewrite
    BlockAndValueMapping bvm;
    bvm.map(op.getOperand(0), input);

    // infer output type
    ArrayRef<int64_t> inputShape =
        input.getType().cast<RankedTensorType>().getShape();
    ArrayRef<int64_t> weightShape =
        weight.getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> newDotOShape({inputShape[0], weightShape[1]});
    RankedTensorType newDotOType = RankedTensorType::get(newDotOShape, dtype);
    auto newDot = cloneAndReplaceResultTypes(rewriter, op, bvm, newDotOType);

    RankedTensorType newReshapeType =
        RankedTensorType::get({newDotOShape[1]}, dtype);
    auto newReshape = rewriter.create<ReshapeOp>(op->getLoc(), newReshapeType,
                                                 newDot->getResult(0));
    auto newBcast = rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), newReshape->getResult(0),
        bcast.broadcast_dimensions());

    return success();
  }

private:
  bool checkBroadcastFirstDimension(BroadcastInDimOp op) const {
    DenseIntElementsAttr bcastDim = op.broadcast_dimensions();
    for (auto it = bcastDim.begin(); it < bcastDim.end(); ++it) {
      if (*it == 0) {
        return false;
      }
    }
    return true;
  }
};

struct SliceMoveDownPattern : public HloMoveDownPattern<mhlo::SliceOp> {
  SliceMoveDownPattern(MLIRContext *context,
                       const llvm::DenseSet<llvm::StringRef> &blocker,
                       bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::SliceOp>(context, blocker, allMultiUser,
                                          multiUser) {}

  LogicalResult matchAndRewrite(mhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> slices;
    SmallVector<Operation *> slice_users;
    Value binary_common_operand;
    bool is_reshape_case = false;

    // match pattern
    patternMatch(op.getOperand(), slices, slice_users, binary_common_operand,
                 is_reshape_case);

    // not possible for fusing, simply fail it
    if (slices.size() <= 1) {
      return failure();
    }

    // rewrite
    return performFuseAndMoveDown(slices, slice_users, binary_common_operand,
                                  is_reshape_case, rewriter);
  }

private:
  void patternMatch(const Value &root, SmallVector<Operation *> &slices,
                    SmallVector<Operation *> &slice_users,
                    Value &binary_common_operand, bool &is_reshape_case) const {
    auto root_ty = root.getType().dyn_cast<RankedTensorType>();
    int64_t rank = root_ty.getRank();
    std::string user_op_name;

    for (auto user : root.getUsers()) {
      // condition 1: must be SliceOp
      mhlo::SliceOp slice = dyn_cast<mhlo::SliceOp>(*user);
      if (!slice) {
        continue;
      }

      // condition 2: slice dim len must be 1
      auto start_attr = slice.start_indices();
      auto limit_attr = slice.limit_indices();
      bool is_all_slice_dim_len_one = true;
      for (int64_t dim_idx = 0; dim_idx < rank; dim_idx++) {
        const int64_t start =
            start_attr.getValues<IntegerAttr>()[dim_idx].getInt();
        const int64_t limit =
            limit_attr.getValues<IntegerAttr>()[dim_idx].getInt();
        if (start + 1 != limit) {
          is_all_slice_dim_len_one = false;
        }
      }
      if (!is_all_slice_dim_len_one) {
        continue;
      }

      auto slice_res = slice.getResult();

      for (auto slice_consumer : slice_res.getUsers()) {
        // condition 3: must be elementwise op or reshape op
        mhlo::ReshapeOp reshape = dyn_cast<mhlo::ReshapeOp>(*slice_consumer);
        if (!slice_consumer->hasTrait<mlir::OpTrait::Elementwise>() &&
            !reshape) {
          continue;
        }

        const auto op_name =
            std::string(slice_consumer->getName().getStringRef());
        // condition 3.1: same user op type, otherwise
        // it might not be possible for fusing
        if (user_op_name.empty()) {
          user_op_name = op_name;
        } else if (user_op_name.compare(op_name)) {
          continue;
        }

        // condition 3.2: reshape op, we may loose this condition
        if (reshape && checkReshapeRemoveFirstNumberOneDimension(reshape)) {
          is_reshape_case = true;
          slices.push_back(user);
          slice_users.push_back(slice_consumer);
          continue;
        }

        // condition 3.3: unary elementwise op
        if (slice_consumer->getNumOperands() == 1) {
          slices.push_back(user);
          slice_users.push_back(slice_consumer);
          continue;
        }

        // condition 3.4: binary elementwise op, for operand, besides SliceOp,
        // share common operand
        bool should_add_binary = true;
        for (auto operand : slice_consumer->getOperands()) {
          if (operand == slice_res) {
            continue;
          }
          if (!binary_common_operand) {
            binary_common_operand = operand;
            continue;
          }
          if (binary_common_operand != operand) {
            should_add_binary = false;
            break;
          }
        }
        if (should_add_binary) {
          slices.push_back(user);
          slice_users.push_back(slice_consumer);
        }
      }
    }
  }

  mhlo::BroadcastInDimOp
  createBroadcastOpForBinaryOperand(SmallVector<Operation *> &slices,
                                    Value &binary_common_operand,
                                    PatternRewriter &rewriter) const {
    auto before_broadcast_type =
        binary_common_operand.getType().cast<RankedTensorType>();
    auto after_broadcast_type =
        slices[0]->getOperand(0).getType().cast<RankedTensorType>();
    llvm::SmallVector<int64_t> broadcast_dimensions(
        before_broadcast_type.getRank());
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    DenseIntElementsAttr new_bcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<long int>(broadcast_dimensions.size())},
            rewriter.getI64Type()),
        broadcast_dimensions);
    return rewriter.create<mhlo::BroadcastInDimOp>(
        rewriter.getUnknownLoc(), after_broadcast_type, binary_common_operand,
        new_bcast_attr);
  }

  auto *createFusedMovedUpOp(SmallVector<Operation *> &slices,
                             SmallVector<Operation *> &slice_users,
                             bool is_reshape_case,
                             mhlo::BroadcastInDimOp &new_broadcast_op,
                             PatternRewriter &rewriter) const {
    auto user = slice_users[0];
    BlockAndValueMapping bvm;
    for (auto operand : user->getOperands()) {
      if (operand == slices[0]->getResult(0)) {
        bvm.map(operand, slices[0]->getOperand(0));
      } else {
        bvm.map(operand, new_broadcast_op.getResult());
      }
    }
    auto new_producer = cloneAndReplaceResultTypes(
        rewriter, user, bvm, slices[0]->getOperandTypes());
    if (is_reshape_case) {
      auto oldOutputTy = slices[0]->getOperand(0).getType();
      Type dtype = oldOutputTy.cast<RankedTensorType>().getElementType();
      ArrayRef<int64_t> oldOutShape =
          oldOutputTy.cast<RankedTensorType>().getShape();
      SmallVector<int64_t> newOutShape;
      for (size_t i = 1; i < oldOutShape.size(); i++) {
        newOutShape.push_back(oldOutShape[i]);
      }
      RankedTensorType newOutType = RankedTensorType::get(newOutShape, dtype);
      new_producer =
          cloneAndReplaceResultTypes(rewriter, user, bvm, newOutType);
    }

    return new_producer;
  }

  LogicalResult performFuseAndMoveDown(SmallVector<Operation *> &slices,
                                       SmallVector<Operation *> &slice_users,
                                       Value &binary_common_operand,
                                       bool is_reshape_case,
                                       PatternRewriter &rewriter) const {
    // step 1: insert Broadcast for common operand
    mhlo::BroadcastInDimOp new_broadcast_op;
    if (binary_common_operand) {
      new_broadcast_op = createBroadcastOpForBinaryOperand(
          slices, binary_common_operand, rewriter);
    }

    // step 2: create new fused user
    auto new_producer = createFusedMovedUpOp(
        slices, slice_users, is_reshape_case, new_broadcast_op, rewriter);

    // step 3: perform move down
    for (size_t i = 0; i < slices.size(); i++) {
      auto op = slices[i];
      mhlo::SliceOp slice = dyn_cast<mhlo::SliceOp>(*op);
      assert(slice);
      auto user = slice_users[i];
      // create slice op
      if (is_reshape_case) {
        llvm::SmallVector<int64_t> old_start_indices;
        llvm::SmallVector<int64_t> old_end_indices;
        llvm::SmallVector<int64_t> old_strides;
        getValuesFromDenseIntElementsAttr(slice.start_indices(),
                                          old_start_indices);
        getValuesFromDenseIntElementsAttr(slice.limit_indices(),
                                          old_end_indices);
        getValuesFromDenseIntElementsAttr(slice.strides(), old_strides);
        // squeeze 1st dim. See Line 674
        llvm::SmallVector<int64_t> new_start_indices(
            old_start_indices.begin() + 1, old_start_indices.end());
        llvm::SmallVector<int64_t> new_end_indices(old_end_indices.begin() + 1,
                                                   old_end_indices.end());
        llvm::SmallVector<int64_t> new_strides(old_strides.begin() + 1,
                                               old_strides.end());
        mlir::Type int64ElementType = rewriter.getI64Type();
        DenseIntElementsAttr start_attr = DenseIntElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(new_start_indices.size())},
                int64ElementType),
            new_start_indices);
        DenseIntElementsAttr end_attr = DenseIntElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(new_end_indices.size())},
                int64ElementType),
            new_end_indices);
        DenseIntElementsAttr stride_attr = DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(new_strides.size())},
                                  int64ElementType),
            new_strides);
        rewriter.replaceOpWithNewOp<mhlo::SliceOp>(
            user, user->getResultTypes(), new_producer->getResult(0),
            start_attr, end_attr, stride_attr);
      } else {
        rewriter.replaceOpWithNewOp<mhlo::SliceOp>(
            user, user->getResultTypes(), new_producer->getResult(0),
            slice.start_indices(), slice.limit_indices(), slice.strides());
      }
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
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());

    // add pattern
    populateHloMoveDownPattern(patterns, {}, allMultiUser, multiUser);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, patterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError(
          "HloMoveDownPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
    funcOp.walk([](ReshapeOp op) { op->removeAttr(kMoveDownDisableKey); });
  }
};
} // namespace

void mlir::populateHloMoveDownPattern(RewritePatternSet &patterns,
                                      const llvm::DenseSet<StringRef> &blocker,
                                      bool allMultiUser, bool multiUser) {
  patterns
      .add<TransposeMoveDownPattern, ReshapeMoveDownPattern,
           BroadcastMoveDownPattern, BroadcastReshapeMoveDownPattern,
           ReshapeBroadcastDotMoveDownPattern, BroadcastBinaryMoveDownPattern
           /*,SliceMoveDownPattern*/>(patterns.getContext(), blocker,
                                      allMultiUser, multiUser);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloMoveDownPass(bool allMultiUser, bool multiUser) {
  return std::make_unique<HloMoveDownPass>(allMultiUser, multiUser);
}
