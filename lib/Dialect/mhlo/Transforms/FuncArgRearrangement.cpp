//===- FuncArgRearrangement.cpp -------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/FuncArgRearrangement.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

class FuncArgRearrangementPass : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = FuncArgRearrangementPass;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuncArgRearrangementPass)

  FuncArgRearrangementPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<FuncArgRearrangementPass>()) {}

  FuncArgRearrangementPass(const FuncArgRearrangementPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  FuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                           const std::string &anchor)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<FuncArgRearrangementPass>()),
        rearrangeBuilder(builder), anchorAttr(anchor) {}

// Note command-line was disable in this pass, due to it using a class to drive
// Please use TestConvertInsertion (test-insert-convert) in command-line
#if 0 
  /// Returns the command-line argument attached to this pass.
   static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("rearrange-func-arg");
  }
  ::llvm::StringRef getArgument() const override { return "rearrange-func-arg"; }

  ::llvm::StringRef getDescription() const override {
    return "Func Arg Rearrangement: pack, reorder args and returns of func";
  }
#endif

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("FuncArgRearrangement");
  }
  ::llvm::StringRef getName() const override { return "FuncArgRearrangement"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<FuncArgRearrangementPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<FuncArgRearrangementPass>(
        *static_cast<const FuncArgRearrangementPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override;

protected:
  FuncArgRearrangerBuilderBase *rearrangeBuilder = nullptr;
  std::string anchorAttr = "";
};

void FuncArgRearrangementPass::runOnOperation() {
  if (rearrangeBuilder == nullptr || anchorAttr.empty())
    return;

  ModuleOp m = getOperation();

  // collect all func
  SmallVector<func::FuncOp> targetFuncs;
  for (auto f : m.getOps<func::FuncOp>()) {
    if (f->hasAttr(anchorAttr)) {
      targetFuncs.push_back(f);
    }
  }

  for (auto f : targetFuncs) {
    SmallVector<Operation *> eraser;

    auto rearrangerPtr = rearrangeBuilder->createFuncArgRearranger(f);
    // skip if nullptr or init failed
    if (rearrangerPtr == nullptr || !rearrangerPtr->init()) {
      continue;
    }

    // 1. Create a new Func
    OpBuilder builder(f);
    // StringRef funcSymName = f.getSymName();
    // auto newFnType = rearrangerPtr->getFunctionType();

    auto newFunc = builder.create<func::FuncOp>(
        f->getLoc(), f.getSymName(), rearrangerPtr->getFunctionType(),
        f.getSymVisibilityAttr());

    // 2. Rewrite Body if Func is non-empty
    if (!f.empty()) {
      // handle args
      auto entry = newFunc.addEntryBlock();
      builder.setInsertionPointToEnd(entry);

      // assign BlockAndValueMapping from oldArg to newVal
      BlockAndValueMapping argBvm;

      auto newArgs = llvm::to_vector(llvm::map_range(
          newFunc.getArguments(),
          [&](const BlockArgument &val) -> Value { return val; }));

      for (unsigned i = 0; i < f.getNumArguments(); ++i) {
        auto maybeToVal =
            rearrangerPtr->getOrCreateOldFromNewFuncArg(builder, i, newArgs);

        if (maybeToVal.hasValue()) {
          argBvm.map(f.getArgument(i), maybeToVal.getValue());
        }
      }

      // clone body by replace oldArgs to newVals
      f.getBody().cloneInto(&newFunc.getBody(), argBvm);
      SmallVector<Operation *> ops;
      for (auto &op : newFunc.getBody().back()) {
        ops.push_back(&op);
      }

      for (auto op : ops) {
        op->moveAfter(&newFunc.getBody().front().back());
      }
      newFunc.getBody().back().erase();

      // handle ReturnOp
      auto oldRet =
          cast<func::ReturnOp>(newFunc.getBody().back().getTerminator());
      builder.setInsertionPoint(oldRet);
      auto oldRetOperands = llvm::to_vector(oldRet.getOperands());
      SmallVector<Value> newRetOperands;
      for (unsigned i = 0; i < newFunc.getNumResults(); ++i) {
        auto newVal = rearrangerPtr->getOrCreateNewFromOldFuncResult(
            builder, i, oldRetOperands);
        newRetOperands.push_back(newVal);
      }

      builder.create<func::ReturnOp>(oldRet.getLoc(), newRetOperands);

      // collect old ReturnOp in eraser
      eraser.push_back(oldRet);
    }

    // 3. Rewrite Call
    auto maybeSymbolUses = f.getSymbolUses(m);
    for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
      if (auto callOp = dyn_cast<func::CallOp>(symbolUse.getUser())) {

        builder.setInsertionPoint(callOp);

        // handle callOp's args
        auto oldArgs = llvm::to_vector(callOp.getOperands());
        SmallVector<Value> newArgs;
        for (unsigned i = 0; i < newFunc.getNumArguments(); ++i) {
          auto newArg =
              rearrangerPtr->getOrCreateNewFromOldFuncArg(builder, i, oldArgs);
          newArgs.push_back(newArg);
        }

        auto newCall =
            builder.create<func::CallOp>(callOp.getLoc(), newFunc, newArgs);

        // handle callOp's results
        auto newCallResults = llvm::to_vector(
            llvm::map_range(newCall.getResults(),
                            [&](const OpResult &val) -> Value { return val; }));

        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          auto maybeToResult = rearrangerPtr->getOrCreateOldFromNewFuncResult(
              builder, i, newCallResults);
          if (maybeToResult.hasValue()) {
            callOp.getResult(i).replaceAllUsesWith(maybeToResult.getValue());
          }
        }

        // collect old callOp in eraser
        eraser.push_back(callOp);
      }
    }

    eraser.push_back(f);

    // erase all ops in eraser
    for (auto op : eraser) {
      op->erase();
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                                     const std::string &anchor) {
  return std::make_unique<FuncArgRearrangementPass>(builder, anchor);
}
