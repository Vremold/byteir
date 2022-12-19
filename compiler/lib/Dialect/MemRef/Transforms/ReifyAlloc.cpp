//===- ReifyAlloc.cpp -----------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/ReifyAlloc.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

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
    addAttrs(newOp, filteredAttrs);

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
  // clang-format off
  patterns.add<ReifyAllocPattern<memref::AllocOp>,
               ReifyAllocPattern<memref::AllocaOp>>(patterns.getContext());
  // clang-format on
}

void ReifyAllocPass::runOnOperation() {
  auto funcOp = getOperation();

  RewritePatternSet patterns(funcOp.getContext());
  populateReifyAllocLikePatterns(patterns);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
    funcOp.emitError(
        "ReifyAllocPass applyPatternsAndFoldGreedily does not converge");
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createReifyAllocPass() {
  return std::make_unique<ReifyAllocPass>();
}
