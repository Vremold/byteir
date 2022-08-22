//===- ResolveShapeConstraint.cpp -----------------------------------C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/Transforms/ResolveShapeConstraint.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "resolve-shape-constraint"

using namespace mlir;

namespace {

struct ResolveShapeConstraintPass
    : public ResolveShapeConstraintBase<ResolveShapeConstraintPass> {
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
    }
  };

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    llvm::EquivalenceClasses<Value, ValueComparator> eqs;
    llvm::SmallVector<Value> constValues;
    funcOp.walk([&eqs, &constValues](shape_ext::MeetOp meetOp) {
      Value lhs = meetOp.getArg0();
      Value rhs = meetOp.getArg1();
      eqs.unionSets(lhs, rhs);
      Operation *lhsOp = lhs.getDefiningOp();
      Operation *rhsOp = rhs.getDefiningOp();
      if (lhsOp && lhsOp->hasTrait<OpTrait::ConstantLike>()) {
        constValues.push_back(lhs);
      }
      if (rhsOp && rhsOp->hasTrait<OpTrait::ConstantLike>()) {
        constValues.push_back(rhs);
      }
    });

    // find constant value of each equivalence class
    llvm::SmallDenseMap<Value, Value> eqsToConstant;
    for (auto constVal : constValues) {
      Value leader = eqs.getLeaderValue(constVal);
      if (!eqsToConstant.count(leader)) {
        eqsToConstant[leader] = constVal;
      } else {
        Value expectVal = eqsToConstant[leader];
        Operation *curOp = constVal.getDefiningOp();
        Operation *expectOp = expectVal.getDefiningOp();
        llvm::Optional<int64_t> curI64Val =
            getLiteralFromConstantLike(curOp->getResult(0));
        llvm::Optional<int64_t> expectI64Val =
            getLiteralFromConstantLike(expectOp->getResult(0));
        if (expectI64Val != curI64Val) {
          std::string msg;
          llvm::raw_string_ostream ss(msg);
          ss << "expect const values in the same shape_ext::meetOp equivalence "
                "class to be the same, got "
             << curI64Val << " while previous value is " << expectI64Val
             << "\n";
          // a bit strange to emit error from the constant op,
          // better to emit from the meetOp but that requries additional mapping
          curOp->emitError(std::move(msg));
        }
      }
    }

    SmallVector<Operation *> toRemoveMeetOp;
    funcOp.walk([&](shape_ext::MeetOp meetOp) {
      Value lhs = meetOp.getArg0();
      Value rhs = meetOp.getArg1();
      Value lhsLeader = eqs.getLeaderValue(lhs);
      Value rhsLeader = eqs.getLeaderValue(rhs);
      assert(lhsLeader == rhsLeader &&
             "operands of a meetOp must be of the same equivalence class");

      if (!eqsToConstant.count(lhsLeader)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "meetOp " << meetOp
                   << " not resolved, no constant equivalence value found\n");
        return WalkResult::advance();
      }

      Value constVal = eqsToConstant[lhsLeader];
      if (constVal != lhs) {
        lhs.replaceAllUsesWith(constVal);
      }
      if (constVal != rhs) {
        rhs.replaceAllUsesWith(constVal);
      }
      toRemoveMeetOp.push_back(meetOp);
      return WalkResult::advance();
    });

    for (auto op : toRemoveMeetOp) {
      op->erase();
    }

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    shape_ext::TieOp::getCanonicalizationPatterns(patterns, ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp->emitError() << "Canonicalize on tie op failed";
      signalPassFailure();
    }
    func::ReturnOp retOp = *funcOp.getOps<func::ReturnOp>().begin();
    // Canonicalize pattern will not modify the funcion type, therefore it need
    // to be set explicitly here.
    funcOp.setType(FunctionType::get(
        funcOp.getContext(),
        //  funcOp.getArgumentTypes(),
        funcOp.getBody().getBlocks().front().getArgumentTypes(),
        retOp.getOperandTypes()));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createResolveShapeConstraintPass() {
  return std::make_unique<ResolveShapeConstraintPass>();
}
