//===- FusionOutling.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/FusionOutlining.h"
#include "PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::mhlo;
using namespace llvm;

namespace {

static std::string GetOutlineFuncitonName(mhlo::FusionOp fusionOp,
                                          unsigned &cnt) {
  StringAttr nameAttr =
      fusionOp->getAttrOfType<StringAttr>(byre::getByreComputeName());
  std::string funcName;

  if (nameAttr == nullptr) {
    funcName = "Unknown" + Twine(cnt++).str();
  } else {
    funcName = nameAttr.getValue().str() + Twine(cnt++).str();
  }

  return funcName;
}

static FuncOp CreateOutlinedFuncOp(mhlo::FusionOp fusionOp,
                                   StringRef funcName) {

  // creat outline function
  auto ctx = fusionOp->getContext();
  SmallVector<Type, 4> inputTypes(fusionOp.getOperandTypes());
  SmallVector<Type, 4> retTypes(fusionOp.getResultTypes());

  OpBuilder opBuilder(fusionOp.getContext());

  FunctionType funcType = FunctionType::get(ctx, inputTypes, retTypes);
  FuncOp funcOp = FuncOp::create(fusionOp.getLoc(), funcName, funcType);
  funcOp.setPrivate();

  // create entry block
  Block *block = funcOp.addEntryBlock();
  BlockAndValueMapping bvm;
  unsigned numArg = funcOp.getNumArguments();
  for (unsigned i = 0; i < numArg; ++i) {
    bvm.map(fusionOp.getOperand(i), funcOp.getArgument(i));
  }

  // clone fusionOp's block into the next block
  fusionOp.fused_computation().cloneInto(&funcOp.getBody(), bvm);
  Block &secondBlock = funcOp.getBody().back();

  // collect all movable ops
  // also collect direct out of scope def
  // LWC: this code has an assumption the FusionOp is
  // generated from fusion pass, which only allows no arg op
  // to be moved to outer scope.
  // TODO: change to it arbitrary scope of def
  SmallVector<Operation *> ops;
  SmallPtrSet<Operation *, 8> opSet;
  for (auto &it : secondBlock.without_terminator()) {
    auto op = &it;
    // all val
    auto num_operand = op->getNumOperands();
    for (unsigned i = 0; i < num_operand; ++i) {
      auto val = op->getOperand(i);
      auto defOp = val.getDefiningOp();

      if (!defOp || opSet.find(defOp) != opSet.end()) {
        // skip if defining op is null or  in the pattern
        continue;
      }

      opBuilder.setInsertionPoint(op);
      auto clonedDefOp = opBuilder.clone(*defOp);
      auto resIdx = FindResultIndex(defOp, val).getValue();

      op->replaceUsesOfWith(val, clonedDefOp->getResult(resIdx));
      opSet.insert(clonedDefOp);
      ops.push_back(clonedDefOp);
    }
    opSet.insert(op);
    ops.push_back(op);
  }

  // move ops
  for (auto op : ops) {
    op->moveBefore(block, block->end());
  }

  // rebuild a new Return
  auto *terminator = secondBlock.getTerminator();
  opBuilder.setInsertionPoint(
      block, block->end()); // the point set at the end of block
  opBuilder.create<mlir::ReturnOp>(terminator->getLoc(),
                                   terminator->getOperands());

  // erase terminator first, and then erase the block
  terminator->erase();
  secondBlock.erase();

  // copy fusionOp's attributes to funcOp
  AddAttrs(funcOp.getOperation(), fusionOp->getAttrs());
  return funcOp;
}

static void RewriteFusionOpToCall(mhlo::FusionOp fusionOp, FuncOp funcOp) {
  // create a call
  OpBuilder opBuilder(fusionOp);
  auto callOp = opBuilder.create<mlir::CallOp>(fusionOp.getLoc(), funcOp,
                                               fusionOp.getOperands());

  // replace all uses of fusionOp by callOp
  unsigned numResult = fusionOp.getNumResults();
  for (unsigned i = 0; i < numResult; ++i) {
    fusionOp.getResult(i).replaceAllUsesWith(callOp.getResult(i));
  }

  // erase fusionOp
  fusionOp.erase();
}

struct FusionOutliningPass : public FusionOutliningBase<FusionOutliningPass> {

  FusionOutliningPass() = default;

  void runOnOperation() override;
};

} // namespace

void FusionOutliningPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  unsigned cnt = 0;

  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    funcOp.walk([&](mhlo::FusionOp fusionOp) {
      auto funcName = GetOutlineFuncitonName(fusionOp, cnt);
      auto outlinedFuncOp = moduleOp.lookupSymbol<FuncOp>(funcName);

      if (outlinedFuncOp == nullptr) {
        outlinedFuncOp = CreateOutlinedFuncOp(fusionOp, funcName);
        moduleOp.insert(funcOp, outlinedFuncOp);

        // Only set the first time

        StringRef byre_compute_name = byre::getByreComputeName();
        SmallVector<NamedAttribute, 8> filteredAttrs(llvm::make_filter_range(
            fusionOp->getAttrs(), [&](NamedAttribute attr) {
              return attr.getName().getValue() != byre_compute_name;
            }));

        AddAttrs(outlinedFuncOp, filteredAttrs);
      }

      RewriteFusionOpToCall(fusionOp, outlinedFuncOp);
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createFusionOutliningPass() {
  return std::make_unique<FusionOutliningPass>();
}
