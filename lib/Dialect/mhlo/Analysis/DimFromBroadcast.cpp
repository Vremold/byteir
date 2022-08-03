//===- DimFromBroadcast.cpp -----------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Analysis/DimFromBroadcast.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace byteir;

namespace {

SmallVector<unsigned> GetGreaterThanOneIdx(ArrayRef<int64_t> array) {
  SmallVector<unsigned> indices;
  for (unsigned i = 0; i < array.size(); ++i) {
    if (array[i] > 1) {
      indices.push_back(i);
    }
  }
  return indices;
}

SmallVector<bool> BroadcastInDimHandleFlag(mhlo::BroadcastInDimOp op,
                                           int64_t rank) {
  auto denseAttr = op.broadcast_dimensions();
  SmallVector<bool> res(rank, true);
  for (int64_t i = 0, e = denseAttr.getNumElements(); i < e; ++i)
    res[denseAttr.getValues<APInt>()[i].getSExtValue()] = false;
  return res;
}

SmallVector<bool> ReshapeHandleFlag(mhlo::ReshapeOp op, int64_t rank,
                                    ArrayRef<int64_t> oup_shape,
                                    DimFlagAnalysis *analysis) {
  SmallVector<bool> res(rank, false);
  Value inp = op.operand();
  auto inp_shaped_type = inp.getType().dyn_cast<ShapedType>();
  if (!inp_shaped_type || !inp_shaped_type.hasRank()) {
    return res;
  }
  // TODO: This is only a conservative check currently. Will not
  // check pattern like X[a*b, c] = mhlo.reshape(Y[a, b, c])
  ArrayRef<int64_t> inp_shape = inp_shaped_type.getShape();
  SmallVector<unsigned> oup_greater_than_one_idx =
      GetGreaterThanOneIdx(oup_shape);
  SmallVector<unsigned> inp_greater_than_one_idx =
      GetGreaterThanOneIdx(inp_shape);
  if (oup_greater_than_one_idx.size() != inp_greater_than_one_idx.size()) {
    return res;
  }
  for (unsigned i = 0; i < oup_greater_than_one_idx.size(); ++i) {
    if (oup_shape[oup_greater_than_one_idx[i]] !=
        inp_shape[inp_greater_than_one_idx[i]]) {
      return res;
    }
  }
  ArrayRef<bool> inp_res = analysis->GetDimFlag(inp);
  for (unsigned i = 0; i < oup_greater_than_one_idx.size(); ++i) {
    res[oup_greater_than_one_idx[i]] = inp_res[inp_greater_than_one_idx[i]];
  }
  return res;
}

SmallVector<bool> UnaryElementwiseHandleFlag(Operation *op,
                                             DimFlagAnalysis *analysis) {
  return analysis->GetDimFlag(op->getOperand(0));
}

SmallVector<bool> BinaryElementwiseHandleFlag(Operation *op,
                                              DimFlagAnalysis *analysis) {
  SmallVector<bool> left = analysis->GetDimFlag(op->getOperand(0));
  SmallVector<bool> right = analysis->GetDimFlag(op->getOperand(1));
  if (left.size() != right.size()) {
    return SmallVector<bool>();
  }
  SmallVector<bool> res(false, left.size());
  for (size_t i = 0; i < left.size(); ++i) {
    res[i] = left[i] && right[i];
  }
  return res;
}
} // namespace

SmallVector<bool> DimFromBroadcast::Compute(Value v) {
  auto shaped_type = v.getType().dyn_cast<ShapedType>();
  if (!shaped_type || !shaped_type.hasRank()) {
    return SmallVector<bool>();
  }
  int64_t cur_rank = shaped_type.getRank();
  ArrayRef<int64_t> cur_shape = shaped_type.getShape();

  Operation *def_op = v.getDefiningOp();
  SmallVector<bool> dim_flag =
      llvm::TypeSwitch<Operation *, SmallVector<bool>>(def_op)
          .Case<mhlo::BroadcastInDimOp>(
              [&](auto op) { return BroadcastInDimHandleFlag(op, cur_rank); })
          .Case<mhlo::ReshapeOp>([&](auto op) {
            return ReshapeHandleFlag(op, cur_rank, cur_shape, analysis_);
          })
          .Case<mhlo::AbsOp, mhlo::CbrtOp, mhlo::CeilOp, mhlo::ConvertOp,
                mhlo::ClzOp, mhlo::CosineOp, mhlo::ExpOp, mhlo::Expm1Op,
                mhlo::FloorOp, mhlo::ImagOp, mhlo::IsFiniteOp, mhlo::LogOp,
                mhlo::Log1pOp, mhlo::LogisticOp, mhlo::NotOp, mhlo::NegOp,
                mhlo::PopulationCountOp, mhlo::RealOp, mhlo::RoundOp,
                mhlo::RsqrtOp, mhlo::SignOp, mhlo::SineOp, mhlo::SqrtOp,
                mhlo::TanhOp>([&](auto op) {
            return UnaryElementwiseHandleFlag(op, analysis_);
          })
          .Case<mhlo::AddOp, mhlo::Atan2Op, mhlo::ComplexOp, mhlo::DivOp,
                mhlo::MaxOp, mhlo::MinOp, mhlo::MulOp, mhlo::PowOp, mhlo::RemOp,
                mhlo::ShiftLeftOp, mhlo::ShiftRightArithmeticOp,
                mhlo::ShiftRightLogicalOp, mhlo::SubtractOp, mhlo::AndOp,
                mhlo::OrOp, mhlo::XorOp>([&](auto op) {
            return BinaryElementwiseHandleFlag(op, analysis_);
          })
          // TODO: Handle more operation types here.
          .Default(
              [&](Operation *) { return SmallVector<bool>(cur_rank, false); });
  return dim_flag;
}