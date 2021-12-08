//===- ToByre.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "byteir/Conversion/Common/FunctionSupport.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h" // LmhloDialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::lmhlo;
using namespace llvm;


template<>
LogicalResult
mlir::ConvertToByrePattern<mlir::lmhlo::ScatterOp>::matchAndRewrite(
  mlir::lmhlo::ScatterOp op,
  typename mlir::lmhlo::ScatterOp::Adaptor adaptor,
  ConversionPatternRewriter& rewriter) const {

  auto found = src_to_callee_.find(op.getOperation()->getName().getStringRef());
  if (found == src_to_callee_.end()) {
    // TODO adding more error message
    return failure();
  }

  // TODO support inplace 
  auto new_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op,
    found->second, adaptor.getOperands());

  return success();
}

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

  auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
      op, found->second, adaptor.getOperands());

  // FIXME: currently only support select on dim0
  compute_op->setAttr("dim", rewriter.getI32IntegerAttr(0));

  return success();
}

namespace {

class ConvertCallOpToByrePattern : public OpConversionPattern<mlir::CallOp> {
public:
  ConvertCallOpToByrePattern(MLIRContext* ctx)
    : OpConversionPattern<mlir::CallOp>(ctx) {}

  LogicalResult
  matchAndRewrite(mlir::CallOp op, mlir::CallOp::Adaptor adaptor, 
    ConversionPatternRewriter& rewriter) const override {

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
      rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(op, nameAttr.getValue(), adaptor.getOperands());

    SmallVector<NamedAttribute> attrs;
    for (auto iter = funcOp->getAttrs().begin(); iter != funcOp->getAttrs().end(); iter++) {
      if (byre::isByreComputeAttr(*iter)) {
        attrs.emplace_back(byre::removeByrePrefix(*iter));
      }
    }

    AddAttrs(computeOp.getOperation(), attrs);

    return success();
  }
};

class ConvertDotOpToByrePattern : public OpConversionPattern<mlir::lmhlo::DotOp> {
public:
  ConvertDotOpToByrePattern(MLIRContext* ctx)
    : OpConversionPattern<mlir::lmhlo::DotOp>(ctx) {}

  LogicalResult
  matchAndRewrite(mlir::lmhlo::DotOp op, 
    mlir::lmhlo::DotOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const override {

    auto dot_dimension_numbers = adaptor.dot_dimension_numbers();
    assert(dot_dimension_numbers.getLhsContractingDimensions().size() == 1);
    assert(dot_dimension_numbers.getRhsContractingDimensions().size() == 1);
    if (dot_dimension_numbers.getLhsBatchingDimensions().size() == 0) {
      // convert to MatmulOp
      auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, "MatmulOp", adaptor.getOperands());

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

      auto compute_op = rewriter.replaceOpWithNewOp<mlir::byre::ComputeOp>(
        op, "BatchMatmulOp", adaptor.getOperands());

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

class ConvertCustomCallOpToByrePattern
  : public OpConversionPattern<lmhlo::CustomCallOp> {
public:
  ConvertCustomCallOpToByrePattern(MLIRContext* ctx)
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

// Main Pass
struct ConvertToByrePass : public ConvertToByreBase<ConvertToByrePass> {

  ConvertToByrePass() : ConvertToByreBase() {
    // TODO: change to loading from outside
    lmhloSupportMap.insert({"lmhlo.add", "AddOp"});
    lmhloSupportMap.insert({ "lmhlo.scatter", "IndexPutOp" });
    lmhloSupportMap.insert({"lmhlo.gather", "IndexSelectOp"});

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

// identify EntryPoint funciton
static void identifyEntryPointFunc(ModuleOp m,
                                   llvm::SmallVector<FuncOp, 4> &collector) {
  // get first entyr func
  for (auto entry : m.getOps<FuncOp>()) {

    // skip non entry-point function or empty func
    if (!isFuncWithEntryPointPlaceholder(entry) && entry.isPublic()) {
      continue;
    }

    collector.push_back(entry);
  }
}

static inline void relocateFuncOpResultsForLmhlo(
    FuncOp func, const llvm::SmallPtrSet<FuncOp, 4> &collectorSet) {
  unsigned idx = func.getNumArguments();

  replcateFuncOpResults(func, [&](mlir::ReturnOp retOp) {
    llvm::SmallPtrSet<mlir::Operation *, 16> removeOps;

    mlir::OpBuilder opBuilder(retOp);

    for (auto retVal : retOp.getOperands()) {

      if (auto allocOp = dyn_cast<memref::AllocOp>(retVal.getDefiningOp())) {
        removeOps.insert(allocOp);
      } else if (auto callOp = dyn_cast<mlir::CallOp>(retVal.getDefiningOp())) {
        // handle call op
        if (callOp.getNumResults() > 0) {
          auto calleeFuncOp = GetFuncOp(callOp);
          if (collectorSet.contains(calleeFuncOp)) {
            opBuilder.setInsertionPoint(callOp);
            SmallVector<Value, 4> oprands(callOp.getOperands());
            oprands.append(callOp.getResults().begin(),
                           callOp.getResults().end());
            mlir::CallOp newCallOp = opBuilder.create<mlir::CallOp>(
                callOp.getLoc(), calleeFuncOp, oprands);
            newCallOp->setAttrs(callOp->getAttrs());
            removeOps.insert(callOp);
          }
        }
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

static inline void relocateFuncOpConstantLikeForLmhlo(FuncOp func,
                                                      unsigned unknownCnt) {

  MLIRContext *ctx = func.getContext();
  SmallVector<Attribute, 16> weightAttrs;

  relocateFuncOpConstantLike(func, "lmhlo.constant", [&](mlir::Operation *op) {
    NamedAttrList attrList;
    auto attr = op->getAttr("name");
    if (attr != nullptr) {
      attrList.append(byre::ByreDialect::getEntryPointFuncArgNameAttrName(),
                      attr);
    } else {
      auto strAttr =
          StringAttr::get(ctx, Twine("UnknowWeight") + Twine(unknownCnt));
      attrList.append(byre::ByreDialect::getEntryPointFuncArgNameAttrName(),
                      strAttr);
    }
    attrList.append(byre::ByreDialect::getEntryPointFuncArgTypeAttrName(),
                    byre::EntryFuncArgTypeAttr::get(
                        op->getContext(), byre::EntryFuncArgType::Weight));
    return std::make_tuple(op->getOperand(0), attrList);
  });
}

static inline void markFuncOpInOutTypeForLmhlo(FuncOp func) {
  auto argTypeAttrName = byre::ByreDialect::getEntryPointFuncArgTypeAttrName();
  for (size_t idx = 0; idx < func.getNumArguments(); ++idx) {
    func.setArgAttr(idx, argTypeAttrName,
                    byre::EntryFuncArgTypeAttr::get(
                        func->getContext(), byre::EntryFuncArgType::Input));
  }
  for (size_t idx = 0; idx < func.getNumResults(); ++idx) {
    func.setResultAttr(idx, argTypeAttrName,
                       byre::EntryFuncArgTypeAttr::get(
                           func->getContext(), byre::EntryFuncArgType::Output));
  }
}

static inline void rewriteByreResultAttrsToFuncResultAttr(FuncOp func) {
  auto resultAttrsName = byre::ByreDialect::getEntryPointFuncResultAttrsName();
  removeAttrPlaceholders(func, {resultAttrsName});
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
  llvm::SmallVector<FuncOp, 4> collector;

  identifyEntryPointFunc(m, collector);

  // early termination if module has no entry point function
  if (collector.size() == 0) {
    return;
  }

  // get a set for
  llvm::SmallPtrSet<FuncOp, 4> collectorSet(collector.begin(), collector.end());

  // insert byre.container_module to module if there is none.
  if (!m->hasAttr(byre::ByreDialect::getContainerModuleAttrName())) {
    m->setAttr(byre::ByreDialect::getContainerModuleAttrName(),
               UnitAttr::get(&ctx));
  }

  unsigned unknownCnt = 0;
  for (auto func : collector) {
    markFuncOpInOutTypeForLmhlo(func);
    rewriteByreResultAttrsToFuncResultAttr(func);
    relocateFuncOpResultsForLmhlo(func, collectorSet);
    if (isFuncWithEntryPointPlaceholder(func)) {
      relocateFuncOpConstantLikeForLmhlo(func, unknownCnt);
    }
    removeAttrPlaceholders(func, attrNames);
    removeArgAttrPlaceholders(func, argAttrNames);
  }

  // Below rewrite Lmhlo ops
  ConversionTarget target(getContext());
  target.addLegalDialect<byre::ByreDialect, memref::MemRefDialect, scf::SCFDialect,
    StandardOpsDialect, ace::AceDialect>();

  target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
  //  target.addLegalDialect<LmhloDialect>();

  target.addDynamicallyLegalDialect<LmhloDialect>([&](Operation *op) {
    auto func = op->getParentOfType<FuncOp>();
    return !isEntryPointFunc(func);
  });

  target.addDynamicallyLegalOp<mlir::CallOp>([&](Operation *op) {
    auto func = op->getParentOfType<FuncOp>();
    return !isEntryPointFunc(func);
  });

  RewritePatternSet patterns(&ctx);
  populateLmhloToByreConversionPatterns(patterns, lmhloSupportMap);

  populateStdToByreConversionPatterns(patterns);

  if (failed(applyFullConversion(m, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

void mlir::populateLmhloToByreConversionPatterns(
  RewritePatternSet& patterns,
  llvm::DenseMap<StringRef, StringRef>& supportMap) {
  // TODO move this from a file
  // TODO use MACRO trick to add patterns
  patterns.add<ConvertToByrePattern<lmhlo::AddOp>,
               ConvertToByrePattern<lmhlo::ScatterOp>,
               ConvertToByrePattern<lmhlo::GatherOp>>(patterns.getContext(),
                                                      supportMap);

  patterns.add<ConvertDotOpToByrePattern,
    ConvertCustomCallOpToByrePattern>(patterns.getContext());
}

void mlir::populateStdToByreConversionPatterns(RewritePatternSet& patterns) {
  patterns.add<ConvertCallOpToByrePattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertToByrePass() {
  return std::make_unique<ConvertToByrePass>();
}
