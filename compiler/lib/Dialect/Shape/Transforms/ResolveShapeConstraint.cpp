//===- ResolveShapeConstraint.cpp -----------------------------------C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/Transforms/ResolveShapeConstraint.h"
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include <queue>

#include "PassDetail.h"

#define DEBUG_TYPE "resolve-shape-constraint"

using namespace mlir;

namespace {

struct ResolveShapeConstraintPass
    : public ResolveShapeConstraintBase<ResolveShapeConstraintPass> {

  using Knowledge = std::pair<Value, int64_t>;

  inline Knowledge getNullKnowledge() { return {nullptr, 0LL}; }

  // get constant dim size from value, return dynamic if failed
  static int64_t getConstIndex(const Value &val) {
    llvm::Optional<int64_t> i64Val = getLiteralFromConstantLike(val);
    int64_t ret = i64Val.value_or(ShapedType::kDynamicSize);
    if (ret == -1) {
      if (auto cOp =
              llvm::dyn_cast_or_null<shape::ConstSizeOp>(val.getDefiningOp())) {
        ret = cOp.getValue().getSExtValue();
      }
    }
    return ret;
  };

  Knowledge deriveKnowledgeFromShapeValue(Value shapeVal, int index,
                                          int64_t dimSize) {
    Operation *op = shapeVal.getDefiningOp();
    if (!op)
      return getNullKnowledge();
    if (auto mhloReshapeOp = llvm::dyn_cast<mhlo::ComputeReshapeShapeOp>(op)) {
      DenseIntElementsAttr shapeAttr;
      if (!matchPattern(mhloReshapeOp.getOperand(1), m_Constant(&shapeAttr)))
        return getNullKnowledge();
      SmallVector<int64_t> shape;
      getValuesFromDenseIntElementsAttr(shapeAttr, shape);
      if (static_cast<int>(shape.size()) <= index)
        getNullKnowledge();
      int dynamicDim = -1;
      int64_t prod = 1;
      for (size_t i = 0, e = shape.size(); i < e; ++i) {
        int64_t dim = shape[i];
        if (!ShapedType::isDynamic(dim)) {
          prod *= dim;
          continue;
        }
        if (dynamicDim != -1)
          return getNullKnowledge();
        dynamicDim = i;
      }
      if (index == dynamicDim)
        return {mhloReshapeOp.getOperand(0), dimSize * prod};
    }
    return getNullKnowledge();
  }

  // derive new Knowledge from current by backward analyzing expressions
  Knowledge deriveKnowledge(Knowledge knowledge) {
    Value dimVal = knowledge.first;
    int64_t dimSize = knowledge.second;
    Operation *op = dimVal.getDefiningOp();
    if (!op)
      return getNullKnowledge();
    if (auto indexCastOp = llvm::dyn_cast<arith::IndexCastOp>(op))
      return {indexCastOp.getIn(), dimSize};

    if (auto extractOp = llvm::dyn_cast<tensor::ExtractOp>(op)) {
      if (extractOp.getIndices().size() != 1u)
        return getNullKnowledge();
      Value indexValue = extractOp.getIndices()[0];
      int64_t index = getConstIndex(indexValue);
      if (index < 0)
        return getNullKnowledge();
      Value tensor = extractOp.getTensor();
      return deriveKnowledgeFromShapeValue(tensor, index, dimSize);
    }

    if (auto numElementsOp = llvm::dyn_cast<shape::NumElementsOp>(op)) {
      Value shapeVal = numElementsOp.getShape();
      Operation *shapeOp = shapeVal.getDefiningOp();
      if (auto fromElementsOp =
              llvm::dyn_cast<tensor::FromElementsOp>(shapeOp)) {
        int64_t dynamicDim = -1;
        int64_t prod = 1;
        for (const auto it : llvm::enumerate(fromElementsOp.getElements())) {
          int64_t dim = getConstIndex(it.value());
          if (!ShapedType::isDynamic(dim)) {
            prod *= dim;
            continue;
          }
          if (dynamicDim != -1)
            return getNullKnowledge();
          dynamicDim = it.index();
        }
        if (dimSize % prod != 0)
          return getNullKnowledge();
        if (dynamicDim != -1)
          return {fromElementsOp.getElements()[dynamicDim], dimSize / prod};
      }
    }

    if (auto mulOp = llvm::dyn_cast<shape::MulOp>(op)) {
      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();
      int64_t lhsDimSize = getConstIndex(lhs);
      int64_t rhsDimSize = getConstIndex(rhs);
      if (ShapedType::isDynamic(lhsDimSize) &&
          ShapedType::isDynamic(rhsDimSize))
        return getNullKnowledge();
      if (ShapedType::isDynamic(lhsDimSize)) {
        std::swap(lhs, rhs);
        std::swap(lhsDimSize, rhsDimSize);
      }
      if (dimSize % lhsDimSize == 0)
        return {rhs, dimSize / lhsDimSize};
    }
    if (auto addOp = llvm::dyn_cast<shape::AddOp>(op)) {
      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();
      int64_t lhsDimSize = getConstIndex(lhs);
      int64_t rhsDimSize = getConstIndex(rhs);
      if (ShapedType::isDynamic(lhsDimSize) &&
          ShapedType::isDynamic(rhsDimSize))
        return getNullKnowledge();
      if (ShapedType::isDynamic(lhsDimSize)) {
        std::swap(lhs, rhs);
        std::swap(lhsDimSize, rhsDimSize);
      }
      if (dimSize - lhsDimSize > 0)
        return {rhs, dimSize - lhsDimSize};
    }
    return getNullKnowledge();
  }

  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
    }
  };

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getBody());

    llvm::EquivalenceClasses<Value, ValueComparator> eqs;
    std::queue<Knowledge> knowledgeQueue;
    llvm::SmallDenseMap<int64_t, Value> dimSizeToValue;
    llvm::SmallDenseMap<Value, int64_t> valueToDimSize;

    auto getOrCreateDimSize = [&](int64_t dimSize) -> Value {
      if (!dimSizeToValue.count(dimSize)) {
        dimSizeToValue[dimSize] =
            builder.create<arith::ConstantIndexOp>(funcOp->getLoc(), dimSize);
      }
      return dimSizeToValue[dimSize];
    };

    auto replaceValueWithConst = [&](Value value, Value constValue) {
      if (value == constValue)
        return;
      value.replaceAllUsesWith(constValue);
    };
    auto applyKnowledge = [&](Knowledge knowledge) -> LogicalResult {
      if (valueToDimSize.count(knowledge.first)) {
        if (valueToDimSize[knowledge.first] != knowledge.second)
          funcOp.emitError()
              << knowledge.first << " set dim size failed, try to set to "
              << knowledge.second << " while previous is "
              << valueToDimSize[knowledge.first] << "\n";
        return failure();
      }
      LLVM_DEBUG(llvm::dbgs() << "apply knowledge: [" << knowledge.first
                              << "] = [" << knowledge.second << "]\n");
      valueToDimSize[knowledge.first] = knowledge.second;
      return success();
    };

    funcOp.walk([&](shape_ext::MeetOp meetOp) {
      Value lhsVal = meetOp.getArg0();
      Value rhsVal = meetOp.getArg1();
      int64_t lhsDim = getConstIndex(lhsVal);
      int64_t rhsDim = getConstIndex(rhsVal);
      bool lhsDyn = ShapedType::isDynamic(lhsDim);
      bool rhsDyn = ShapedType::isDynamic(rhsDim);
      // only join dynamic dim
      if (lhsDyn && rhsDyn)
        eqs.unionSets(lhsVal, rhsVal);
      // memorize already appeared const dim, to avoid unnecessary op creation
      if (!lhsDyn)
        dimSizeToValue[lhsDim] = lhsVal;
      if (!rhsDyn)
        dimSizeToValue[rhsDim] = rhsVal;
      // add knowledges
      if (lhsDyn && !rhsDyn)
        knowledgeQueue.emplace(lhsVal, rhsDim);
      if (rhsDyn && !lhsDyn)
        knowledgeQueue.emplace(rhsVal, lhsDim);
    });

    llvm::SmallDenseSet<Value> seenLeaders;
    while (!knowledgeQueue.empty()) {
      Knowledge knowledge = knowledgeQueue.front();
      knowledgeQueue.pop();
      if (failed(applyKnowledge(knowledge)))
        continue;
      // try to derive new knowledge from current
      Knowledge newKnowledge = deriveKnowledge(knowledge);
      if (newKnowledge.first != nullptr)
        knowledgeQueue.emplace(newKnowledge);

      Value leader = eqs.getOrInsertLeaderValue(knowledge.first);
      if (seenLeaders.count(leader))
        continue;
      seenLeaders.insert(leader);
      auto leaderIt = eqs.findValue(leader);
      for (auto it = eqs.member_begin(leaderIt); it != eqs.member_end(); ++it) {
        if (*it != knowledge.first)
          knowledgeQueue.emplace(*it, knowledge.second);
      }
    }
    for (const auto &it : valueToDimSize)
      replaceValueWithConst(it.first, getOrCreateDimSize(it.second));

    funcOp.walk([&](shape_ext::MeetOp meetOp) { meetOp->erase(); });

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