//===- ReifyAlloc.cpp ------------------------------------------*--- C++
//-*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/ReifyAlloc.h"
#include "PassDetail.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

template <typename OpTy>
struct ReifyAllocPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {

    bool valid = llvm::any_of(opTy->getOperands(), [&](Value val) {
      return isa_and_nonnull<arith::ConstantIndexOp>(val.getDefiningOp());
    });

    if (!valid)
      return failure();

    SmallVector<int64_t> newShape;
    SmallVector<Value> newOperands;

    auto oldMemRef =
        opTy->getResult(0).getType().template dyn_cast<MemRefType>();
    reifyAllocLikeShapeAndOperands(oldMemRef.getShape(), opTy->getOperands(),
                                   newShape, newOperands);

    MemRefType newMmRefType =
        MemRefType::get(newShape, oldMemRef.getElementType(),
                        oldMemRef.getLayout(), oldMemRef.getMemorySpace());

    auto newOp =
        rewriter.replaceOpWithNewOp<OpTy>(opTy, newMmRefType, newOperands);

    // cloned all attrs, except alloc specific attribute, operand_segment_sizes
    SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(opTy->getAttrs(), [&](NamedAttribute attr) {
          return attr.getName().getValue() != "operand_segment_sizes";
        }));
    AddAttrs(newOp, filteredAttrs);

    return success();
  }
};

struct ReifyAllocPass : public ReifyAllocBase<ReifyAllocPass> {
public:
  ReifyAllocPass() = default;
  void runOnOperation() override;
};

} // namespace

void mlir::reifyAllocLikeShapeAndOperands(ArrayRef<int64_t> oldShape,
                                          ValueRange oldOperands,
                                          SmallVectorImpl<int64_t> &newShape,
                                          SmallVectorImpl<Value> &newOperands) {

  unsigned idx = 0;
  for (size_t i = 0; i < oldShape.size(); ++i) {
    auto dim = oldShape[i];
    if (dim >= 0) {
      newShape.push_back(dim);
    } else {
      auto val = oldOperands[idx++];
      if (auto cOp =
              dyn_cast_or_null<arith::ConstantIndexOp>(val.getDefiningOp())) {
        newShape.push_back(cOp.value());
        ;
      } else {
        newOperands.push_back(val);
      }
    }
  }
}

void mlir::populateReifyAllocLikePatterns(RewritePatternSet &patterns) {
  patterns.add<ReifyAllocPattern<memref::AllocOp>,
               ReifyAllocPattern<memref::AllocaOp>>(patterns.getContext());
}

void ReifyAllocPass::runOnOperation() {
  auto funcOp = getOperation();

  RewritePatternSet patterns(funcOp.getContext());
  populateReifyAllocLikePatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp.emitError(
        "ReifyAllocPass applyPatternsAndFoldGreedily does not converge");
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createReifyAllocPass() {
  return std::make_unique<ReifyAllocPass>();
}
