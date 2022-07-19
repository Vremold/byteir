//===- ResolveShapeConstraint.cpp -----------------------------------C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/Transforms/ResolveShapeConstraint.h"
#include "PassDetail.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct ResolveShapeConstraintPass
    : public ResolveShapeConstraintBase<ResolveShapeConstraintPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    SmallVector<Operation *> toRemoveMeetOp;

    funcOp.walk([&](shape_ext::MeetOp meetOp) {
      Value lhs = meetOp.getArg0();
      Value rhs = meetOp.getArg1();
      Operation *lhsOp = lhs.getDefiningOp();
      Operation *rhsOp = rhs.getDefiningOp();
      bool isLhsConstLike = (lhsOp) && lhsOp->hasTrait<OpTrait::ConstantLike>();
      bool isRhsConstLike = (rhsOp) && rhsOp->hasTrait<OpTrait::ConstantLike>();

      if (!isLhsConstLike && !isRhsConstLike) {
        return WalkResult::advance();
      }

      if (isLhsConstLike && isRhsConstLike) {
        llvm::Optional<int64_t> lhsValue =
            getLiteralFromConstantLike(lhsOp->getResults()[0]);
        llvm::Optional<int64_t> rhsValue =
            getLiteralFromConstantLike(rhsOp->getResults()[0]);
        if (lhsValue != rhsValue) {
          std::string msg;
          llvm::raw_string_ostream ss(msg);
          ss << "constant operands not equal: " << lhsValue
             << " != " << rhsValue;
          meetOp->emitOpError(std::move(msg));
        }
      } else if (isLhsConstLike) {
        rhs.replaceAllUsesWith(lhs);
      } else if (isRhsConstLike) {
        lhs.replaceAllUsesWith(rhs);
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
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp->emitError() << "Canonicalize on tie op failed";
      signalPassFailure();
    }
    func::ReturnOp retOp = *funcOp.getOps<func::ReturnOp>().begin();
    // Canonicalize pattern will not modify the funcion type, therefore it need
    // to be set explicitly here.
    funcOp.setType(FunctionType::get(funcOp.getContext(),
                                     funcOp.getArgumentTypes(),
                                     retOp.getOperandTypes()));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createResolveShapeConstraintPass() {
  return std::make_unique<ResolveShapeConstraintPass>();
}
