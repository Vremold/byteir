//===- BoundedShapeInference.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/BoundedShapeInference.h"
#include "./PassDetail.h"
#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "bounded-shape-infer"

namespace {

LogicalResult constructNewArgumentTypes(func::FuncOp funcOp,
                                        SmallVectorImpl<Type> &newArgumentTypes,
                                        SmallVectorImpl<Type> &newFuncArgTypes,
                                        OpBuilder &builder) {
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    Type origType = funcOp.getArgumentTypes()[i];

    auto origRankedType = origType.dyn_cast<RankedTensorType>();
    if (!origRankedType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Argument " << i << "is not of type RankedTensorType.\n");
      return failure();
    }

    if (origRankedType.hasStaticShape()) {
      newArgumentTypes.push_back(origType);
      newFuncArgTypes.push_back(origType);
      continue;
    }

    auto boundedShapeAttr =
        funcOp.getArgAttrOfType<ArrayAttr>(i, getBoundedShapeAttrName());
    if (!boundedShapeAttr) {
      LLVM_DEBUG(llvm::dbgs() << "Argument " << i
                              << "doesn't have either static shape or "
                                 "bounded shape attribute.\n");
      return failure();
    }

    SmallVector<int64_t> boundedShape = llvm::to_vector(
        llvm::map_range(boundedShapeAttr.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));

    if (int64_t(boundedShape.size()) != origRankedType.getRank()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Argument " << i << "'s rank: " << origRankedType.getRank()
                 << ", is not equal to bounded shape's rank: "
                 << boundedShape.size() << "\n");
      return failure();
    }

    auto typeWithEncoding = RankedTensorType::get(
        origRankedType.getShape(), origRankedType.getElementType(),
        builder.getDictionaryAttr(
            {builder.getNamedAttr(getBoundedShapeAttrName(),
                                  builder.getI64ArrayAttr(boundedShape))}));
    newFuncArgTypes.push_back(typeWithEncoding);
    Type newType = origRankedType.clone(boundedShape);
    newArgumentTypes.push_back(newType);
  }
  return success();
}

struct BoundedShapeInferencePass
    : public BoundedShapeInferenceBase<BoundedShapeInferencePass> {

  BoundedShapeInferencePass()
      : BoundedShapeInferenceBase<
            BoundedShapeInferencePass>::BoundedShapeInferenceBase() {
    registerAllMhloInferBoundedReturnTypes();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp);

    SmallVector<Type> newArgumentTypes;
    SmallVector<Type> newFuncArgTypes;
    if (failed(constructNewArgumentTypes(funcOp, newArgumentTypes,
                                         newFuncArgTypes, builder))) {
      return;
    }

    // Construct new FuncOp
    StringRef funcSymName = funcOp.getSymName();
    std::string newFuncSymName = "_bounded_shape_infer_" + funcSymName.str();
    auto newFnType =
        builder.getFunctionType(newArgumentTypes, funcOp.getResultTypes());
    auto newFuncOp = builder.create<func::FuncOp>(funcOp->getLoc(),
                                                  newFuncSymName, newFnType);

    BlockAndValueMapping emptyBvm;
    funcOp.getBody().cloneInto(&newFuncOp.getBody(), emptyBvm);
    for (auto it : zip(newArgumentTypes, newFuncOp.getArguments())) {
      std::get<1>(it).setType(std::get<0>(it));
    }

    //  Run shape inference on the new created function op
    if (failed(runShapeInference(newFuncOp, /*isBoundedShapeInfer=*/true))) {
      return;
    }

    // Set bounded shape attr according to the inferred results
    std::vector<Operation *> originalOps;
    SmallVector<Type> newFuncRetTypes;
    funcOp.walk([&](Operation *op) { originalOps.push_back(op); });
    unsigned opIndex = 0;
    func::ReturnOp returnOp;
    newFuncOp.walk([&](Operation *op) {
      opIndex++;
      // set bounded type attr for the original op
      Operation *originalOp = originalOps[opIndex - 1];
      for (auto it : llvm::enumerate(
               llvm::zip(originalOp->getResults(), op->getResults()))) {
        auto originalType =
            std::get<0>(it.value()).getType().dyn_cast<ShapedType>();
        if (!originalType || originalType.hasStaticShape())
          continue;
        auto newType = std::get<1>(it.value()).getType().dyn_cast<ShapedType>();
        if (!newType || !newType.hasRank())
          continue;
        std::string boundedShapeAttrName =
            getBoundedShapeAttrName().str() + std::to_string(it.index());
        ArrayRef<int64_t> shape = newType.getShape();
        auto resultValue = std::get<0>(it.value());
        auto tx = resultValue.getType().dyn_cast<RankedTensorType>();
        if (tx) {
          auto typeWithEncoding = RankedTensorType::get(
              tx.getShape(), tx.getElementType(),
              builder.getDictionaryAttr({builder.getNamedAttr(
                  getBoundedShapeAttrName(), builder.getI64ArrayAttr(shape))}));
          resultValue.setType(typeWithEncoding);
        } else {
          originalOp->setAttr(boundedShapeAttrName,
                              builder.getI64ArrayAttr(shape));
        }
      }
      if (isa<func::ReturnOp>(originalOp)) {
        returnOp = dyn_cast<func::ReturnOp>(originalOp);
      }
    });

    // must find return op
    newFuncRetTypes = llvm::to_vector(returnOp.getOperandTypes());

    auto newFuncType =
        FunctionType::get(&getContext(), newFuncArgTypes, newFuncRetTypes);
    funcOp.setType(newFuncType);
    for (auto elem :
         llvm::zip(funcOp.front().getArguments(), newFuncArgTypes)) {
      std::get<0>(elem).setType(std::get<1>(elem));
    }

    // Erase the auxiliary function op at the end
    newFuncOp->erase();
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createBoundedShapeInferencePass() {
  return std::make_unique<BoundedShapeInferencePass>();
}
