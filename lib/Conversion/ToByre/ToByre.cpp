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
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h" // LmhloDialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::lmhlo;
using namespace llvm;

void mlir::populateLmhloToByreConversionPatterns(
    RewritePatternSet &patterns,
    llvm::DenseMap<StringRef, StringRef> &supportMap) {
  // TODO move this from a file
  // TODO use MACRO trick to add patterns
  patterns.add<ConvertToByrePattern<lmhlo::AddOp>>(patterns.getContext(),
                                                   supportMap);
  patterns.add<ConvertToByrePattern<lmhlo::CustomCallOp>>(patterns.getContext());
  patterns.add<ConvertToByrePattern<lmhlo::DotOp>>(patterns.getContext());
}

void mlir::populateStdToByreConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertToByrePattern<mlir::CallOp>>(patterns.getContext());
}

namespace {

// Main Pass
struct ConvertToByrePass : public ConvertToByreBase<ConvertToByrePass> {

  ConvertToByrePass() : ConvertToByreBase() {
    // TODO: change to loading from outside
    lmhloSupportMap.insert({"lmhlo.add", "AddOp"});
    // lmhlo.dot will convert to MatmulOp and BatchMatmulOp
    // lmhloSupportMap.insert({ "lmhlo.dot", "MatmulOp" });

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

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertToByrePass() {
  return std::make_unique<ConvertToByrePass>();
}
