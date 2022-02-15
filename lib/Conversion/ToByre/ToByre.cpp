//===- ToByre.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToByre/ToByre.h"
#include "../PassDetail.h"
#include "byteir/Conversion/Common/FunctionSupport.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h" // LmhloDialect
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <functional>
#include <string>

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::lmhlo;
using namespace mlir::mhlo;
using namespace llvm;

template <>
mlir::LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::GatherOp>::matchAndRewrite(
    mlir::lmhlo::GatherOp op, typename mlir::lmhlo::GatherOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  auto start_indices = op.start_indices();
  auto start_indices_ty = start_indices.getType().cast<ShapedType>();
  if (!start_indices_ty.hasRank()) {
    return rewriter.notifyMatchFailure(op, "unranked start_indices");
  }

  auto operand = op.operand();
  auto operand_ty = operand.getType().cast<ShapedType>();
  if (!operand_ty.hasRank()) {
    return rewriter.notifyMatchFailure(op, "unranked operand");
  }

  int64_t index_vector_dim = start_indices_ty.getRank();

  auto dimension_numbers = op.dimension_numbers();
  if (dimension_numbers.getIndexVectorDim() != index_vector_dim) {
    return rewriter.notifyMatchFailure(
        op, "index_vector_dim not last dimension of start_indices");
  }

  // Index select only works across a single dimension.
  if (start_indices_ty.getShape().empty() || start_indices_ty.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "start_indices index vector dimension not 1");
  }

  // Only support the default case for start_index_map.
  if (dimension_numbers.getStartIndexMap().size() != 1 ||
      dimension_numbers.getStartIndexMap()[0] != 0) {
    return rewriter.notifyMatchFailure(op, "start_index_map != [0]");
  }

  auto result_ty = op.output().getType().dyn_cast<ShapedType>();
  if (!result_ty) {
    return rewriter.notifyMatchFailure(op, "unranked result");
  }

  // Offset dimensions should be the defaults.
  if (dimension_numbers.getOffsetDims().size() !=
      result_ty.getRank() - index_vector_dim) {
    return rewriter.notifyMatchFailure(
        op, "offset_dims.size not operand rank minus index_vector_dim");
  }

  for (auto it : llvm::enumerate(dimension_numbers.getOffsetDims())) {
    if ((it.index() + index_vector_dim) != it.value()) {
      return rewriter.notifyMatchFailure(
          op, "offset_dims != [index_vector_dim, result.rank)");
    }
  }

  for (auto it : llvm::enumerate(op.slice_sizes().getValues<APInt>())) {
    // First shape value must be 1.
    if (it.index() == 0) {
      if (it.value().getSExtValue() != 1) {
        return rewriter.notifyMatchFailure(op, "slice_size[0] != 1");
      }
      continue;
    }

    // The op needs to index the entire slice for each other dimension.
    if (it.value().getSExtValue() != operand_ty.getDimSize(it.index())) {
      return rewriter.notifyMatchFailure(
          op, "slice_size doesn't match operand dimension");
    }
  }

  if (dimension_numbers.getCollapsedSliceDims().size() != 1 ||
      dimension_numbers.getCollapsedSliceDims()[0] != 0) {
    return rewriter.notifyMatchFailure(op, "collapsed_slice_dims != [0]");
  }

  auto key = getByreKey(found->second, op->getOperandTypes(), appendArgTypes);

  auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, key, adaptor.getOperands());

  // FIXME: currently only support select on dim0
  compute_op->setAttr("dim", rewriter.getI32IntegerAttr(0));

  return success();
}

template<>
LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::ScatterOp>::matchAndRewrite(
  mlir::lmhlo::ScatterOp op,
  typename mlir::lmhlo::ScatterOp::Adaptor adaptor,
  ConversionPatternRewriter& rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  // check wthether scatter supported
  Region& region = op.update_computation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return rewriter.notifyMatchFailure(op, "unsupported region in scatter");
  }

  auto& block = region.front();
  if (!IsBlockSingleAdd(&block)) {
    return rewriter.notifyMatchFailure(op, "unsupported block in scatter");
  }

  auto key = getByreKey(found->second, op->getOperandTypes(), appendArgTypes);

  // TODO support inplace 
  auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op,
    key, adaptor.getOperands());

  // FIXME: currently only support select on dim0
  new_op->setAttr("dim", rewriter.getI32IntegerAttr(0));

  return success();
}

template<>
LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::SliceOp>::matchAndRewrite(
  mlir::lmhlo::SliceOp op,
  typename mlir::lmhlo::SliceOp::Adaptor adaptor,
  ConversionPatternRewriter& rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  // check whether Slice is applicable for Alias
  if (!isSplatValue(op.strides(), 1)) {
    return rewriter.notifyMatchFailure(op, "unsupported strides of slice");
  }

  auto output = adaptor.getOperands()[1];
  auto shape = output.getType().cast<MemRefType>().getShape();
  auto start_indices = op.start_indices();
  int64_t num_start = start_indices.getNumElements();
  // check high dim of shape is 1
  if (num_start > 1) {
    for (int64_t i = 0; i < num_start-1; ++i) {
      if (shape[i] != 1) {
        return rewriter.notifyMatchFailure(op, "unsupport shape of slice");
      }
    }
  }

  // get last element of start_indices
  int64_t last_start = start_indices.getValues<int64_t>()[num_start - 1];

  // if output is an arg, use copy
  if (adaptor.getOperands()[1].getDefiningOp() == nullptr) {
    auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::CopyOp>(
      op, adaptor.getOperands()[0], adaptor.getOperands()[1]);

    new_op->setAttr("offset", rewriter.getI32IntegerAttr(last_start));
    return success();
  }

  auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
    op, found->second, adaptor.getOperands());

  new_op->setAttr("offset", rewriter.getI32IntegerAttr(last_start));

  if (adaptor.getOperands()[0].getDefiningOp() == nullptr) {
    new_op->setAttr("arg_alias", rewriter.getUnitAttr());
  }

  return success();
}

template <>
LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::ReshapeOp>::matchAndRewrite(
    mlir::lmhlo::ReshapeOp op, typename mlir::lmhlo::ReshapeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  // If both args, replace it with copy
  if (adaptor.getOperands()[0].getDefiningOp() == nullptr && 
      adaptor.getOperands()[1].getDefiningOp() == nullptr) {
    auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::CopyOp>(
      op, adaptor.getOperands()[0], adaptor.getOperands()[1]);

    return success();
  } 

  bool is_arg_alias = false;
  SmallVector<Value> operands;

  // if output is an arg, swap in and out
  if (adaptor.getOperands()[1].getDefiningOp() == nullptr) {
    operands = { adaptor.getOperands()[1], adaptor.getOperands()[0]};
    is_arg_alias = true;
  } else if (adaptor.getOperands()[0].getDefiningOp() == nullptr) {
    operands = adaptor.getOperands();
    is_arg_alias = true;
  } else {
    operands = adaptor.getOperands();
  }

  auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, found->second, operands);

  new_op->setAttr("offset", rewriter.getI32IntegerAttr(0));

  if (is_arg_alias) {
    new_op->setAttr("arg_alias", rewriter.getUnitAttr());
  }

  return success();
}

namespace {

class ConvertCallOpToByrePattern : public OpConversionPattern<mlir::CallOp> {
public:
  ConvertCallOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<mlir::CallOp>(ctx) {}

  LogicalResult
  matchAndRewrite(mlir::CallOp op, mlir::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto funcOp = GetFuncOp(op);
    if (funcOp == nullptr) {
      return failure();
    }

    StringAttr nameAttr =
        funcOp->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (nameAttr == nullptr) {
      return failure();
    }

    mlir::byre::ComputeOp computeOp =
        rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
            op, nameAttr.getValue(), adaptor.getOperands());

    SmallVector<NamedAttribute> attrs;
    for (auto iter = funcOp->getAttrs().begin();
         iter != funcOp->getAttrs().end(); iter++) {
      if (byre::isByreComputeAttr(*iter)) {
        attrs.emplace_back(byre::removeByrePrefix(*iter));
      }
    }

    AddAttrs(computeOp.getOperation(), attrs);

    return success();
  }
};

class ConvertDotOpToByrePattern : public OpConversionPattern<mlir::lmhlo::DotOp> {
private:
  bool appendArgTypes;
public:
  ConvertDotOpToByrePattern(MLIRContext* ctx, bool appendTypes)
    : OpConversionPattern<mlir::lmhlo::DotOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::lmhlo::DotOp op, 
    mlir::lmhlo::DotOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const override {

    auto dot_dimension_numbers = adaptor.dot_dimension_numbers();
    assert(dot_dimension_numbers.getLhsContractingDimensions().size() == 1);
    assert(dot_dimension_numbers.getRhsContractingDimensions().size() == 1);
    if (dot_dimension_numbers.getLhsBatchingDimensions().size() == 0) {
      // convert to MatmulOp
      auto key = getByreKey("MatmulOp", op->getOperandTypes(), appendArgTypes);

      auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, adaptor.getOperands());

      // append attribute 'lhs_contracting_dimension' and
      // 'rhs_contracting_dimension'
      int64_t lhs_contracting_dimension =
        dot_dimension_numbers.getLhsContractingDimensions()[0];
      int64_t rhs_contracting_dimension =
        dot_dimension_numbers.getRhsContractingDimensions()[0];
      compute_op->setAttr("lhs_contracting_dimension",
        rewriter.getI64IntegerAttr(lhs_contracting_dimension));
      compute_op->setAttr("rhs_contracting_dimension",
        rewriter.getI64IntegerAttr(rhs_contracting_dimension));
    }
    else {
      // convert to BatchMatmulOp
      SmallVector<int64_t> batching_dimensions;
      for (int64_t i = 0, e = op.output().getType().cast<ShapedType>().getRank();
        i < e - 2; i++) {
        batching_dimensions.push_back(i);
      }
      if (!dot_dimension_numbers.getLhsBatchingDimensions().equals(
        batching_dimensions) ||
        !dot_dimension_numbers.getRhsBatchingDimensions().equals(
          batching_dimensions)) {
        return op->emitOpError()
          << "can not handle unregular batching_dimensions";
      }

      auto key = getByreKey("BatchMatmulOp", op->getOperandTypes(), appendArgTypes);
      auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, adaptor.getOperands());

      // append attributes of batching and contracting dimensions
      int64_t lhs_contracting_dimension =
        dot_dimension_numbers.getLhsContractingDimensions()[0];
      int64_t rhs_contracting_dimension =
        dot_dimension_numbers.getRhsContractingDimensions()[0];
      auto lhs_batching_dimensions =
        dot_dimension_numbers.getLhsBatchingDimensions();
      auto rhs_batching_dimensions =
        dot_dimension_numbers.getRhsBatchingDimensions();
      compute_op->setAttr("lhs_contracting_dimension",
        rewriter.getI64IntegerAttr(lhs_contracting_dimension));
      compute_op->setAttr("rhs_contracting_dimension",
        rewriter.getI64IntegerAttr(rhs_contracting_dimension));
      compute_op->setAttr("lhs_batching_dimensions",
        rewriter.getI64ArrayAttr(lhs_batching_dimensions));
      compute_op->setAttr("rhs_batching_dimensions",
        rewriter.getI64ArrayAttr(rhs_batching_dimensions));
    }
    return success();
  }
};

class ConvertConvOpToByrePattern
    : public OpConversionPattern<mlir::lmhlo::ConvOp> {
private:
  bool appendArgTypes;

public:
  ConvertConvOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mlir::lmhlo::ConvOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::lmhlo::ConvOp op, mlir::lmhlo::ConvOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    NamedAttrList attrs;
    HandleConvAttribute(attrs, op, rewriter);
    auto key = getByreKey("ConvOp", op->getOperandTypes(), appendArgTypes);
    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, adaptor.getOperands());
    AddAttrs(compute_op.getOperation(), attrs.getAttrs());
    return success();
  }
};

class ConvertCustomCallOpToByrePattern
  : public OpConversionPattern<lmhlo::CustomCallOp> {
public:
  ConvertCustomCallOpToByrePattern(MLIRContext* ctx, bool /*appendArgTypes*/)
    : OpConversionPattern<lmhlo::CustomCallOp>(ctx) {}

  LogicalResult
    matchAndRewrite(lmhlo::CustomCallOp op,
      lmhlo::CustomCallOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    mlir::DictionaryAttr dict_attr;
    auto backend_config = op.backend_config();
    if (!backend_config.empty()) {
      auto attrs = mlir::parseAttribute(backend_config, op->getContext());
      if (!attrs || !attrs.isa<mlir::DictionaryAttr>())
        return failure();
      dict_attr = attrs.cast<mlir::DictionaryAttr>();
    }

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, op.call_target_name(), adaptor.getOperands());
    if (dict_attr) {
      NamedAttrList originAttrs = compute_op->getAttrs();
      originAttrs.append(dict_attr);
      compute_op->setAttrs(originAttrs);
    }

    return success();
  }
};

class ConvertSelectAndScatterOpToByrePattern
  : public OpConversionPattern<lmhlo::SelectAndScatterOp> {
private:
  bool appendArgTypes;
public:
  ConvertSelectAndScatterOpToByrePattern(MLIRContext* ctx, bool appendTypes)
    : OpConversionPattern<lmhlo::SelectAndScatterOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(lmhlo::SelectAndScatterOp op, 
                  lmhlo::SelectAndScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // check whether SelectAndScatterOp support
    if (op.select().getBlocks().size() != 1 ) {
      return rewriter.notifyMatchFailure(op, "unsupported select in select_and_scatter");
    }

    if (op.scatter().getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported scatter in select_and_scatter");
    }

    auto& selectBlock = op.select().front();
    if (selectBlock.getOperations().size() != 2 || 
        !isa<mlir::mhlo::ReturnOp>(selectBlock.getTerminator())) {
      return rewriter.notifyMatchFailure(op, "unsupported block in select of select_and_scatter");
    }

    if (selectBlock.getNumArguments() != 2 ||
        !selectBlock.getArgument(0).getType().isa<TensorType>() ||
        !selectBlock.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported block's arg in select of select_and_scatter");
    }

    auto& scatterBlock = op.scatter().front();
    if (scatterBlock.getOperations().size() != 2 ||
        !isa<mlir::mhlo::ReturnOp>(scatterBlock.getTerminator())) {
      return rewriter.notifyMatchFailure(op, "unsupported block in scatter of select_and_scatter");
    }

    if (scatterBlock.getNumArguments() != 2 ||
        !scatterBlock.getArgument(0).getType().isa<TensorType>() ||
        !scatterBlock.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported block's arg in scatter of select_and_scatter");
    }

    // check whether valid PoolingGrad
    // only support MaxPoolingGrad now
    if (auto compare = dyn_cast<mhlo::CompareOp>(selectBlock.front())) {
      if (compare.comparison_direction() != "GE" || 
          compare->getOperand(0) != selectBlock.getArgument(0) ||
          compare->getOperand(1) != selectBlock.getArgument(1)) {
        return rewriter.notifyMatchFailure(
          op, "unsupported comparison_direction in select of select_and_scatter");
      }
    } else {
      return rewriter.notifyMatchFailure(
        op, "unsupported ops in select of select_and_scatter");
    }

    if (!isa<mhlo::AddOp>(scatterBlock.front()) || 
        scatterBlock.front().getOperand(0) != scatterBlock.getArgument(0) ||
        scatterBlock.front().getOperand(1) != scatterBlock.getArgument(1)) {
      return rewriter.notifyMatchFailure(
        op, "unsupported ops in scatter of select_and_scatter");
    }

    // TODO: more SelectAndScatterOp supported
    std::string poolingGradOp = "PoolMaxGradOp";

    SmallVector<Value, 2> operands{adaptor.operand(), adaptor.source(), adaptor.out()};
    SmallVector<Type, 2> operandTypes{adaptor.operand().getType(),
                                      adaptor.source().getType(),
                                      adaptor.out().getType()};

    auto key = getByreKey(poolingGradOp, operandTypes, appendArgTypes);

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, key, operands);

    AddAttrs(compute_op, op->getAttrs());
    return success();
  }

};

class ConvertReduceOpToByrePattern
    : public OpConversionPattern<lmhlo::ReduceOp> {
private: 
  bool appendArgTypes;
public:
  ConvertReduceOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<lmhlo::ReduceOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(lmhlo::ReduceOp op, lmhlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.inputs().size() != 1 || adaptor.out().size() != 1 ||
        adaptor.init_values().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "batched reductions is not supported yet");
    }
    // check whether reduce supported
    Region &region = op.body();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported region in reduce");
    }
    auto &block = region.front();
    if (block.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op, "unsupported block in reduce");
    }
    // check block args
    if (block.getNumArguments() != 3 ||
        !block.getArgument(0).getType().isa<MemRefType>() ||
        !block.getArgument(1).getType().isa<MemRefType>() ||
        !block.getArgument(2).getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported block's arg in reduce");
    }

    // check block body
    auto ret_op = block.getTerminator();
    if (!isa<mlir::lmhlo::TerminatorOp>(ret_op)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported terminator in reduce's block");
    }

    auto reduce_computation = &block.front();
    // only support ReduceSum now
    if (!isa<lmhlo::AddOp>(reduce_computation)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in reduce_computation of reduce");
    }
    if (reduce_computation->getOperand(0) != block.getArgument(0) ||
        reduce_computation->getOperand(1) != block.getArgument(1) ||
        reduce_computation->getOperand(2) != block.getArgument(2)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in reduce_computation in reduce");
    }
    // TODO: more ReduceOp supported
    std::string ReduceOp = "ReduceSumOp";

    auto inputShape = adaptor.inputs()[0].getType().dyn_cast<MemRefType>();
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    std::vector<int64_t> dimensions;
    for (auto &&i : op.dimensions()) {
      auto dim = i.getSExtValue();
      if (dim < 0 || dim >= inputShape.getRank()) {
        return rewriter.notifyMatchFailure(op, "invalid reduce dimensions");
      }
      dimensions.push_back(dim);
    }
    std::sort(dimensions.begin(), dimensions.end());
    for (size_t i = 0; i < dimensions.size() - 1; ++i) {
      if (dimensions[i + 1] - dimensions[i] != 1)
        return rewriter.notifyMatchFailure(
            op, "only consecutive dimensions were support");
    }

    if (!llvm::any_of(adaptor.init_values()[0].getUses(), [](OpOperand &use) {
          if (auto constOp =
                  llvm::dyn_cast_or_null<lmhlo::ConstOp>(use.getOwner())) {
            // for ReduceSum initial value must be zero
            return isZeroAttribute(constOp.value());
          }
          return false;
        })) {
      return failure();
    }
    SmallVector<Value, 2> operands{adaptor.inputs()[0], adaptor.out()[0]};
    SmallVector<Type, 2> operandTypes{adaptor.inputs()[0].getType(), 
                                      adaptor.out()[0].getType()};

    auto key = getByreKey(ReduceOp, operandTypes, appendArgTypes);

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, key, operands);

    compute_op->setAttr("dimensions", op.dimensions());

    return success();
  }
};


class ConvertReduceWindowOpToByrePattern
  : public OpConversionPattern<lmhlo::ReduceWindowOp> {
private:
  bool appendArgTypes;
public:
  ConvertReduceWindowOpToByrePattern(MLIRContext* ctx, bool appendTypes)
    : OpConversionPattern<lmhlo::ReduceWindowOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
    matchAndRewrite(lmhlo::ReduceWindowOp op, lmhlo::ReduceWindowOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {

    if (adaptor.inputs().size() != 1 || adaptor.out().size() != 1 ||
      adaptor.init_values().size() != 1) {
      return rewriter.notifyMatchFailure(
        op, "batched reductions is not supported yet");
    }
    // check whether reduce supported
    Region& region = op.body();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported region in reduce_window");
    }
    auto& block = region.front();
    if (block.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op, "unsupported block in reduce_window");
    }
    // check block args
    if (block.getNumArguments() != 3 ||
      !block.getArgument(0).getType().isa<MemRefType>() ||
      !block.getArgument(1).getType().isa<MemRefType>() ||
      !block.getArgument(2).getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op,
        "unsupported block's arg in reduce_window");
    }

    // check block body
    auto ret_op = block.getTerminator();
    if (!isa<mlir::lmhlo::TerminatorOp>(ret_op)) {
      return rewriter.notifyMatchFailure(
        op, "unsupported terminator in reduce's block");
    }

    auto reduce_computation = &block.front();
    // only support ReduceWindowMax now
    if (!isa<lmhlo::MaxOp>(reduce_computation)) {
      return rewriter.notifyMatchFailure(
        op, "unsupported ops in reduce_computation of reduce_window");
    }
    if (reduce_computation->getOperand(0) != block.getArgument(0) ||
      reduce_computation->getOperand(1) != block.getArgument(1) ||
      reduce_computation->getOperand(2) != block.getArgument(2)) {
      return rewriter.notifyMatchFailure(
        op, "unsupported ops in reduce_computation in reduce_window");
    }

    // TODO: more ReduceOp supported
    std::string ReduceWinOp = "PoolMaxOp";

    auto inputShape = adaptor.inputs()[0].getType().dyn_cast<MemRefType>();
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    if (!llvm::any_of(adaptor.init_values()[0].getUses(), [](OpOperand& use) {
      if (auto constOp =
        llvm::dyn_cast_or_null<lmhlo::ConstOp>(use.getOwner())) {
        // for ReduceWindowsMax initial value must be minValue
        return isMinValueAttribute(constOp.value());
      }
      return false;
      })) {
      return failure();
    }
    SmallVector<Value, 2> operands{ adaptor.inputs()[0], adaptor.out()[0] };
    SmallVector<Type, 2> operandTypes{ adaptor.inputs()[0].getType(),
                                      adaptor.out()[0].getType() };

    auto key = getByreKey(ReduceWinOp, operandTypes, appendArgTypes);

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, key, operands);

    compute_op->setAttr("window_dimensions", op.window_dimensions());

    if (op.window_strides().hasValue()) {
      compute_op->setAttr("window_strides", op.window_strides().getValue());
    } 

    if (op.padding().hasValue()) {
      compute_op->setAttr("padding", op.padding().getValue());
    }

    return success();
  }
};

class ConvertConstOpToByrePattern : public OpConversionPattern<lmhlo::ConstOp> {
public:
  ConvertConstOpToByrePattern(MLIRContext *ctx, bool /*appendArgTypes*/)
      : OpConversionPattern<lmhlo::ConstOp>(ctx) {}

  LogicalResult
  matchAndRewrite(lmhlo::ConstOp op, lmhlo::ConstOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.value().isSplat())
      return failure();

    auto alloc = op.output().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, "FillOp", adaptor.getOperands());

    compute_op->setAttr("value", op.value());

    return success();
  }
};

// Main Pass
struct ConvertToByrePass : public ConvertToByreBase<ConvertToByrePass> {
  ConvertToByrePass(bool appendArgTypes) : ConvertToByreBase() {
    this->appendArgTypes = appendArgTypes;

    // TODO: change to loading from outside
    lmhloSupportMap.insert({"lmhlo.add", "AddOp"});
    lmhloSupportMap.insert({"lmhlo.scatter", "IndexPutOp" });
    lmhloSupportMap.insert({"lmhlo.gather", "IndexSelectOp"});
    lmhloSupportMap.insert({"lmhlo.reshape", "AliasOp"});
    lmhloSupportMap.insert({"lmhlo.slice", "AliasOp" });

    // insert attrNames
    attrNames.push_back(byre::ByreDialect::getEntryPointFunctionAttrName());
    argAttrNames.push_back(
        byre::ByreDialect::getEntryPointFuncArgNameAttrName());
    argAttrNames.push_back(
        byre::ByreDialect::getEntryPointFuncArgTypeAttrName());
  }

  void runOnOperation() override;

  llvm::DenseMap<StringRef, StringRef> lmhloSupportMap;
  llvm::SmallVector<StringRef, 4> attrNames;
  llvm::SmallVector<StringRef, 4> argAttrNames;
  llvm::SmallVector<StringRef, 4> resultAttrNames;
};

static bool isFuncWithEntryPointPlaceholder(FuncOp func) {
  return func->hasAttr(
      getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()));
}

static bool isEntryPointFunc(FuncOp func) {
  return func->hasAttr(ByreDialect::getEntryPointFunctionAttrName());
}

static bool isRewritablePrivateFunc(FuncOp func) {
  // check support attribute
  return func.isPrivate() &&
         func->hasAttr(getByreComputeName());
}

// identify EntryPoint funciton
static void identifyEntryPointFuncAndCalls(
  ModuleOp m,
  llvm::SmallVector<FuncOp, 4>& entries,
  llvm::SmallVector<mlir::CallOp, 16>& calls,
  llvm::SmallVector<FuncOp, 16>& removeFuncs) {
  // get first entry func

  llvm::SmallPtrSet<Operation*, 16> callSet;

  for (auto func : m.getOps<FuncOp>()) {
    // skip non entry-point function or empty func
    if (!isFuncWithEntryPointPlaceholder(func) || func.isPrivate()) {
      continue;
    }
    entries.push_back(func);

    for (auto callOp : func.getOps<mlir::CallOp>()) {
      auto calleeFuncOp = GetFuncOp(callOp);
      if (isRewritablePrivateFunc(calleeFuncOp) &&
          !callSet.contains(callOp)) {
        calls.push_back(callOp);
        callSet.insert(callOp);
        removeFuncs.push_back(calleeFuncOp);
      }
    }
  }
}

static inline void relocateFuncOpResultsForLmhlo(
    FuncOp func) {
  unsigned idx = func.getNumArguments();
  replcateFuncOpResults(func, [&](mlir::ReturnOp retOp) {
    llvm::SmallPtrSet<mlir::Operation *, 16> removeOps;
    mlir::OpBuilder opBuilder(retOp);
    for (auto retVal : retOp.getOperands()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(retVal.getDefiningOp())) {
        removeOps.insert(allocOp);
      }
      retVal.replaceAllUsesExcept(func.getArgument(idx++), retOp);
    }

    // build and remove return first
    opBuilder.setInsertionPoint(retOp);
    opBuilder.create<mlir::ReturnOp>(retOp.getLoc());
    retOp.erase();

    // remove all remove ops
    for (auto op : removeOps) {
      op->erase();
    }

  });
}

static inline void rewriteCallOpsForFuncOp (
  ArrayRef<mlir::CallOp> calls){

  for (auto callOp : calls) {
    if (callOp.getNumResults() == 0) {
      continue;
    }
    mlir::OpBuilder opBuilder(callOp);
    SmallVector<Value, 4> oprands(callOp.getOperands());

    // change result to alloc
    for (auto r : callOp.getResults()) {

      auto alloc = opBuilder.create<memref::AllocOp>(
        callOp.getLoc(), r.getType().dyn_cast<MemRefType>());
      r.replaceAllUsesExcept(alloc.getResult(), callOp);
      oprands.push_back(alloc.getResult());
    }

    mlir::CallOp newCallOp = opBuilder.create<mlir::CallOp>(
        callOp.getLoc(), callOp.getCalleeAttr(), TypeRange(), oprands);
    newCallOp->setAttrs(callOp->getAttrs());

  }

  // remove all remove ops
  for (auto op : calls) {
    op->erase();
  }
}

static inline void relocateFuncOpConstantLikeForLmhlo(FuncOp func,
                                                      unsigned unknownCnt) {

  MLIRContext *ctx = func.getContext();
  SmallVector<Attribute, 16> weightAttrs;

  lmhlo::ConstOp op;

  relocateFuncOpConstantLike(
      func,
      [&](mlir::Operation *op) {
        if (auto constant = dyn_cast<lmhlo::ConstOp>(op)) {
          return !(constant.value().isSplat());
        }
        return false;
      },
      [&](mlir::Operation *op) {
        NamedAttrList attrList;
        auto attr = op->getAttr("name");
        if (attr != nullptr) {
          attrList.append(byre::ByreDialect::getEntryPointFuncArgNameAttrName(),
                          attr);
        } else {
          auto strAttr =
              StringAttr::get(ctx, Twine("UnknowWeight") + Twine(unknownCnt++));
          attrList.append(byre::ByreDialect::getEntryPointFuncArgNameAttrName(),
                          strAttr);
        }
        attrList.append(byre::ByreDialect::getEntryPointFuncArgTypeAttrName(),
                        byre::EntryFuncArgTypeAttr::get(
                            op->getContext(), byre::EntryFuncArgType::Weight));
        return std::make_tuple(op->getOperand(0), attrList);
      });
}

static inline void markFuncOpInOutTypeForLmhlo(FuncOp func, unsigned inputCnt,
                                               unsigned outputCnt) {
  auto argTypeAttrName = byre::ByreDialect::getEntryPointFuncArgTypeAttrName();
  auto argNameAttrName = byre::ByreDialect::getEntryPointFuncArgNameAttrName();
  auto context = func->getContext();
  for (size_t idx = 0; idx < func.getNumArguments(); ++idx) {
    func.setArgAttr(
        idx, argNameAttrName,
        StringAttr::get(context, Twine("Input") + Twine(inputCnt++)));
    func.setArgAttr(idx, argTypeAttrName,
                    byre::EntryFuncArgTypeAttr::get(
                        context, byre::EntryFuncArgType::Input));
  }
  for (size_t idx = 0; idx < func.getNumResults(); ++idx) {
    func.setResultAttr(
        idx, argNameAttrName,
        StringAttr::get(context, Twine("Output") + Twine(outputCnt++)));
    func.setResultAttr(idx, argTypeAttrName,
                       byre::EntryFuncArgTypeAttr::get(
                           context, byre::EntryFuncArgType::Output));
  }
}

static inline void rewriteByreResultAttrsToFuncResultAttr(FuncOp func) {
  auto resultAttrsName = byre::ByreDialect::getEntryPointFuncResultAttrsName();
  removeAttrPlaceholders(func, { resultAttrsName });
  if (auto result_attrs_attr =
    func->getAttrOfType<mlir::ArrayAttr>(resultAttrsName)) {
    auto new_result_attrs = result_attrs_attr.getValue();
    if (func.getNumResults() != new_result_attrs.size())
      return;
    for (size_t i = 0; i < new_result_attrs.size(); ++i) {
      if (auto new_result_attrs_dict =
        new_result_attrs[i].dyn_cast_or_null<DictionaryAttr>()) {
        NamedAttrList origin_attrs = func.getResultAttrs(i);
        origin_attrs.append(new_result_attrs_dict.getValue());
        func.setResultAttrs(i, origin_attrs.getDictionary(func->getContext()));
      }
    }
    func->removeAttr(resultAttrsName);
  }
}

void ConvertToByrePass::runOnOperation() {

  ModuleOp m = getOperation();
  MLIRContext &ctx = getContext();
  llvm::SmallVector<FuncOp, 4> entryCollector;
  llvm::SmallVector<mlir::CallOp, 16> callCollector;
  llvm::SmallVector<FuncOp, 16> removeFuncCollector;

  identifyEntryPointFuncAndCalls(m, 
    entryCollector, callCollector,
    removeFuncCollector);

  // early termination if module has no entry point function
  if (entryCollector.size() == 0) {
    return;
  }

  // insert byre.container_module to module if there is none.
  if (!m->hasAttr(byre::ByreDialect::getContainerModuleAttrName())) {
    m->setAttr(byre::ByreDialect::getContainerModuleAttrName(),
               UnitAttr::get(&ctx));
  }

  // rewrite private calls
  rewriteCallOpsForFuncOp(callCollector);

  unsigned unknownWeightCnt = 0;
  unsigned inputCnt = 0, outputCnt = 0;
  for (auto func : entryCollector) {
    // Note: In this process we will give all arguments and results of given
    // func a unique `argName`, all arguments would be treated as argType::Input
    // and all results would be treated as argType::Output. But if argument of
    // func was with attribute placholders `argName` and `argType`, it will
    // overwrite those two attributes later.
    markFuncOpInOutTypeForLmhlo(func, inputCnt, outputCnt);

    rewriteByreResultAttrsToFuncResultAttr(func);

    relocateFuncOpResultsForLmhlo(func);
 
    if (isFuncWithEntryPointPlaceholder(func)) {
      relocateFuncOpConstantLikeForLmhlo(func, unknownWeightCnt);
    }

    removeAttrPlaceholders(func, attrNames);
    removeArgAttrPlaceholders(func, argAttrNames);
  }

  // Below rewrite Lmhlo ops
  ConversionTarget target(getContext());
  target.addLegalDialect<byre::ByreDialect, memref::MemRefDialect, scf::SCFDialect,
    StandardOpsDialect, ace::AceDialect>();

  target.addLegalOp<ModuleOp, FuncOp, mlir::ReturnOp>();

  target.addDynamicallyLegalDialect<LmhloDialect>([&](Operation *op) {
    auto func = op->getParentOfType<FuncOp>();
    return !isEntryPointFunc(func);
  });

  target.addDynamicallyLegalOp<mlir::CallOp>([&](Operation *op) {
    auto func = op->getParentOfType<FuncOp>();
    return !isEntryPointFunc(func);
  });

  RewritePatternSet patterns(&ctx);
  populateLmhloToByreConversionPatterns(patterns, lmhloSupportMap, appendArgTypes);

  populateStdToByreConversionPatterns(patterns);

  if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
    signalPassFailure();
  }

  for (auto func : removeFuncCollector) {
    func->erase();
  }

  // remove unused fill op
  m.walk([&](byre::ComputeOp op) {
    if (op.callee() == "FillOp") {
      auto value = op->getOperand(0);
      if (value.hasOneUse() && value.getDefiningOp<memref::AllocOp>()) {
        op->erase();
      }
    }
  });
}

} // namespace

void mlir::populateLmhloToByreConversionPatterns(
  RewritePatternSet& patterns,
  llvm::DenseMap<StringRef, StringRef>& supportMap,
  bool appendArgTypes) {
  // clang-format off
  // TODO move this from a file
  // TODO use MACRO trick to add patterns
  patterns.add<ConvertToByrePattern<lmhlo::AddOp>,
               ConvertToByrePattern<lmhlo::GatherOp>,
               ConvertToByrePattern<lmhlo::ReshapeOp>,
               ConvertToByrePattern<lmhlo::ScatterOp>,
               ConvertToByrePattern<lmhlo::SliceOp>>(patterns.getContext(),
                                                     supportMap, 
                                                     appendArgTypes);

  patterns.add<ConvertConstOpToByrePattern,
               ConvertCustomCallOpToByrePattern,
               ConvertDotOpToByrePattern,
               ConvertConvOpToByrePattern,
               ConvertReduceOpToByrePattern,
               ConvertReduceWindowOpToByrePattern, 
               ConvertSelectAndScatterOpToByrePattern>(
      patterns.getContext(), appendArgTypes);
  // clang-format on
}

void mlir::populateStdToByreConversionPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertCallOpToByrePattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertToByrePass(bool appendArgTypes) {
  return std::make_unique<ConvertToByrePass>(appendArgTypes);
}
