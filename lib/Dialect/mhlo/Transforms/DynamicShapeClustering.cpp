//===- DynamicShapeClustering.cpp -----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/DynamicShapeClustering.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "dynamic-shape-clustering"

namespace {

struct DynamicSourceAnalysis {
  DynamicSourceAnalysis(Operation *op);

  void calPostDomsByTie() {
    operation_->walk(
        [&](shape_ext::TieOp tieOp) { postDomByTieMem_[tieOp] = true; });
    operation_->walk([&](Operation *op) { calPostDomByTieRecursively(op); });
  }

  bool calPostDomByTieRecursively(Operation *op) {
    auto it = postDomByTieMem_.find(op);
    if (it != postDomByTieMem_.end())
      return it->second;

    if (op->getNumResults() == 0) {
      postDomByTieMem_[op] = false;
      return false;
    }

    for (Value res : op->getResults()) {
      bool allDomed = llvm::all_of(res.getUsers(), [&](Operation *user) {
        return calPostDomByTieRecursively(user);
      });
      if (!allDomed) {
        postDomByTieMem_[op] = false;
        return false;
      }
    }

    postDomByTieMem_[op] = true;
    return true;
  }

  void calDynamicSource() {
    std::vector<shape_ext::TieOp> tieOps;
    DenseMap<Value, DenseSet<Value>> dynamicSourcesMem;

    operation_->walk([&](shape_ext::TieOp tieOp) { tieOps.push_back(tieOp); });
    for (shape_ext::TieOp tieOp : tieOps) {
      Value v = tieOp.getValue();
      DenseSet<Value> sources;
      for (Value dimSize : tieOp.getDims()) {
        DenseSet<Value> &dimSources =
            calDynSrcRecursively(dimSize, dynamicSourcesMem);
        for (Value source : dimSources) {
          sources.insert(source);
        }
      }
      dynamicSources_[v] = sources;
    }
  }

  DenseSet<Value> &
  calDynSrcRecursively(Value v,
                       DenseMap<Value, DenseSet<Value>> &dynamicSourcesMem) {
    auto it = dynamicSourcesMem.find(v);
    if (it != dynamicSourcesMem.end())
      return it->second;

    Operation *defOp = v.getDefiningOp();
    // This requires no mhlo ops in shape reification implementation
    if (nullptr == defOp || llvm::isa<mhlo::MhloDialect>(defOp->getDialect())) {
      dynamicSourcesMem[v] = {v};
      return dynamicSourcesMem[v];
    }

    DenseSet<Value> res;
    for (Value inp : defOp->getOperands()) {
      for (Value source : calDynSrcRecursively(inp, dynamicSourcesMem)) {
        res.insert(source);
      }
    }
    dynamicSourcesMem[v] = res;
    return dynamicSourcesMem[v];
  }

  void removeTieOps() {
    std::vector<Operation *> ops;
    operation_->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { ops.push_back(op); });

    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      if (postDomByTieMem_[*it]) {
        (*it)->erase();
      }
    }
  }

  void print(llvm::raw_ostream &os) {
    os << "=================== DynamicSourceAnalysis Printer "
          "=====================\n";
    for (auto it : dynamicSources_) {
      Value v = it.first;
      os << "Sources of " << v << "\n";
      for (Value source : it.second)
        os << source << "\n";
      os << "\n";
    }
  }

  bool isDynamicSource(Value v) {
    auto it = dynamicSources_.find(v);
    if (it == dynamicSources_.end())
      return false;
    if (it->second.size() == 1 && v == *(it->second.begin())) {
      return true;
    }
    return false;
  }

  DenseMap<Operation *, bool> postDomByTieMem_;
  DenseMap<Value, DenseSet<Value>> dynamicSources_;
  Operation *operation_;
};

DynamicSourceAnalysis::DynamicSourceAnalysis(Operation *operation)
    : operation_(operation) {
  calPostDomsByTie();
  calDynamicSource();
}

struct DynamicShapeClusteringPass
    : public DynamicShapeClusteringBase<DynamicShapeClusteringPass> {

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symTable(moduleOp);
    SmallVector<func::FuncOp> funcOps;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      funcOps.push_back(funcOp);
    }

    for (auto funcOp : funcOps) {
      DynamicSourceAnalysis dynSrcAnalysis(funcOp);
      // TODO: make removeTieOps optional or as an ioslated pass.
      dynSrcAnalysis.removeTieOps();

      auto isFusibleCandidate = [&](Operation *op) {
        if (llvm::isa<tensor::DimOp, tensor::FromElementsOp, shape::ShapeOfOp,
                      mhlo::DynamicBroadcastInDimOp, shape::BroadcastOp,
                      shape::CstrBroadcastableOp, shape::AssumingOp,
                      tensor::ExtractOp>(op) ||
            op->hasTrait<OpTrait::ConstantLike>())
          return true;

        if (llvm::isa<mhlo::MhloDialect>(op->getDialect())) {
          for (Value operand : op->getOperands()) {
            if (auto shapeType = operand.getType().dyn_cast<ShapedType>()) {
              if (!shapeType.hasStaticShape())
                return true;
            }
          }
        }

        return false;
      };

      auto isFusibleStart = [&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (dynSrcAnalysis.isDynamicSource(operand)) {
            return true;
          }
        }

        if (op->hasTrait<OpTrait::ConstantLike>())
          return true;

        return false;
      };

      auto isFusibleTrigger = [&](Operation *op) { return true; };

      auto isFusibleWith = [&](Operation *target, Operation *start) {
        if (llvm::isa<tensor::DimOp, tensor::FromElementsOp, shape::ShapeOfOp,
                      mhlo::DynamicBroadcastInDimOp, shape::BroadcastOp,
                      shape::CstrBroadcastableOp, shape::AssumingOp,
                      tensor::ExtractOp>(start))
          return true;

        DenseSet<Value> targetSources;
        DenseSet<Value> startSources;
        for (Value v : target->getResults()) {
          for (Value s : dynSrcAnalysis.dynamicSources_[v]) {
            targetSources.insert(s);
          }
        }
        for (Value v : start->getResults()) {
          for (Value s : dynSrcAnalysis.dynamicSources_[v]) {
            startSources.insert(s);
          }
        }

        if (!startSources.empty() &&
            target->hasTrait<OpTrait::ConstantLike>()) {
          return true;
        }

        llvm::set_intersect(targetSources, startSources);
        return !targetSources.empty();
      };

      for (auto &block : funcOp.getBlocks()) {
        ReplicateDefiningOp(&block, [](Operation *op) {
          return op->hasTrait<OpTrait::ConstantLike>();
        });
      }

      ProducerFusionPlanner planner(funcOp, isFusibleCandidate, isFusibleStart,
                                    isFusibleTrigger, isFusibleWith);
      planner.run();

      const MhloFusionPlan &plan = planner.getFusionPlan();

      std::string namePrefix = funcOp.getSymName().str() + "_sub_";
      int idx = 0;
      for (auto it = plan.rbegin(); it != plan.rend(); ++it) {
        auto &pattern = *it;
        if (pattern.size() == 1 &&
            pattern[0]->hasTrait<OpTrait::ConstantLike>())
          continue;
        OpBuilder b(pattern.back());
        std::string name = namePrefix + std::to_string(idx++);
        func::FuncOp subFnOp = createFuncOpFromPattern(b, name, pattern);
        symTable.insert(subFnOp);
      }
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createDynamicShapeClusteringPass() {
  return std::make_unique<DynamicShapeClusteringPass>();
}
