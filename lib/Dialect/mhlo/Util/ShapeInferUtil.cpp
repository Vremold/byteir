//===- ShapeInferUtil.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Dialect/mhlo/BoundedShapes/Register.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "shape-infer-util"

namespace {

using ResultShapes = SmallVector<ArrayRef<int64_t>, 1>;

LogicalResult checkAndSetTypes(Operation *op,
                               const ResultShapes &inferredShapes) {
  auto results = op->getResults();

  // check
  for (auto it : llvm::zip(results, inferredShapes)) {
    auto result = std::get<0>(it);
    auto resultShape = result.getType().dyn_cast<ShapedType>();
    if (!resultShape || !resultShape.hasRank())
      continue;
    auto inferredShape = std::get<1>(it);

    if (resultShape.getRank() != int64_t(inferredShape.size())) {
      op->emitError()
          << "Found rank mismatch during shape inferring, previous is "
          << resultShape.getRank() << ", inferred is " << inferredShape.size()
          << "\n";
      return failure();
    }

    for (auto dimIt : llvm::zip(resultShape.getShape(), inferredShape)) {
      int64_t lDim = std::get<0>(dimIt);
      int64_t rDim = std::get<1>(dimIt);
      if (lDim > 0 && rDim > 0 && lDim != rDim) {
        op->emitError()
            << "Found dimension mismatch during shape inferring, previous is "
            << lDim << ", inferred is " << rDim << "\n";
        return failure();
      }
    }
  }

  // set
  for (auto it : llvm::zip(results, inferredShapes)) {
    Value result = std::get<0>(it);
    auto shapedType = result.getType().dyn_cast<ShapedType>();
    if (!shapedType)
      continue;
    result.setType(shapedType.clone(std::get<1>(it)));
  }

  return success();
}

LogicalResult inferBoundedShapeUsingRegistry(Operation *op) {
  InferBoundedReturnTypes inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc = inferBoundedReturnTypes(customCall.call_target_name());
  } else {
    inferFunc = inferBoundedReturnTypes(op->getName().getStringRef());
  }

  if (nullptr == inferFunc) {
    return success();
  }

  SmallVector<Type> resultShapeTypes;
  LogicalResult inferStatus =
      inferFunc(op->getContext(), op->getLoc(), op->getOperands(),
                op->getAttrDictionary(), op->getRegions(), resultShapeTypes);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs() << "Registered InferBoundedReturnTypes failed for "
                            << *op << "\n");
    return success();
  }

  ResultShapes resultShapes =
      llvm::to_vector(llvm::map_range(resultShapeTypes, [](Type t) {
        if (auto shape = t.dyn_cast<ShapedType>())
          return shape.getShape();
        return ArrayRef<int64_t>(llvm::None);
      }));
  return checkAndSetTypes(op, resultShapes);
}

LogicalResult inferShapeUsingSameOperandsAndResultShapeTrait(Operation *op) {
  Type staticShapedType = nullptr;
  for (Type t : op->getOperandTypes()) {
    if (t.cast<ShapedType>().hasStaticShape()) {
      staticShapedType = t;
      break;
    }
  }
  if (!staticShapedType) {
    LLVM_DEBUG(llvm::dbgs() << "There's no operand type with static shape in "
                            << *op << "\n");
    return success();
  }

  ResultShapes resultShapes(op->getNumResults(),
                            staticShapedType.cast<ShapedType>().getShape());
  return checkAndSetTypes(op, resultShapes);
}

LogicalResult
inferShapeUsingInferShapedTypeOpInterface(InferShapedTypeOpInterface op) {
  SmallVector<ShapedTypeComponents> resultShapeComps;
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  ValueShapeRange operandsShapeRange(operands);
  LogicalResult inferStatus = op.inferReturnTypeComponents(
      op->getContext(), op->getLoc(), operandsShapeRange,
      op->getAttrDictionary(), op->getRegions(), resultShapeComps);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs()
               << "InferReturnTypeComponents failed for " << op << "\n");
    return success();
  }

  ResultShapes resultShapes = llvm::to_vector(
      llvm::map_range(resultShapeComps, [](ShapedTypeComponents comp) {
        return comp.getDims();
      }));

  return checkAndSetTypes(op, resultShapes);
}

LogicalResult inferShapeUsingInferTypeOpInterface(InferTypeOpInterface op) {
  SmallVector<Type> resultShapeTypes;
  LogicalResult inferStatus = op.inferReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getRegions(), resultShapeTypes);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs() << "InferReturnTypes failed for " << op << "\n");
    return success();
  }
  for (auto it : llvm::zip(op->getResults(), resultShapeTypes)) {
    Value result = std::get<0>(it);
    result.setType(std::get<1>(it));
  }

  ResultShapes resultShapes =
      llvm::to_vector(llvm::map_range(resultShapeTypes, [](Type t) {
        if (auto shape = t.dyn_cast<ShapedType>())
          return shape.getShape();
        return ArrayRef<int64_t>(llvm::None);
      }));
  return checkAndSetTypes(op, resultShapes);
}

LogicalResult inferResultShapes(Operation *op, bool isBoundedShapeInfer) {
  if (isBoundedShapeInfer && failed(inferBoundedShapeUsingRegistry(op))) {
    return failure();
  } else if (op->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
    // Note: some ops has InferShapedTypeOpInterface but it will return
    // failure() directly in the implementation, therefore
    // SameOperandsAndResultShape trait should be checked before checking
    // InferShapedTypeOpInterface
    return inferShapeUsingSameOperandsAndResultShapeTrait(op);
  } else if (auto shapeInferOp = dyn_cast<InferShapedTypeOpInterface>(op)) {
    return inferShapeUsingInferShapedTypeOpInterface(shapeInferOp);
  } else if (auto shapeInferOp = dyn_cast<InferTypeOpInterface>(op)) {
    return inferShapeUsingInferTypeOpInterface(shapeInferOp);
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// runShapeInference
//===----------------------------------------------------------------------===//

// TODO: supported nested function call
LogicalResult mlir::runShapeInference(FuncOp funcOp, bool isBoundedShapeInfer) {
  bool interrupted =
      funcOp
          ->walk([&](Operation *op) {
            if (failed(inferResultShapes(op, isBoundedShapeInfer))) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted();

  if (interrupted)
    return failure();

  func::ReturnOp retOp = *funcOp.getOps<func::ReturnOp>().begin();
  funcOp.setType(FunctionType::get(
      funcOp.getContext(), funcOp.getArgumentTypes(), retOp.getOperandTypes()));

  return success();
}

//===----------------------------------------------------------------------===//
// ReifyReturnTypeShapes Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<ReifyReturnTypeShapes> &
getReifyReturnTypeShapesRegistry() {
  static llvm::StringMap<ReifyReturnTypeShapes> reifyReturnTypeShapesRegistry;
  return reifyReturnTypeShapesRegistry;
}

/// Register the given ReifyReturnTypeShapes function.
static void
registerReifyReturnTypeShapes(StringRef name,
                              const ReifyReturnTypeShapes &function) {
  auto &reifyReturnTypeShapesRegistry = getReifyReturnTypeShapesRegistry();
  if (reifyReturnTypeShapesRegistry.find(name) !=
      reifyReturnTypeShapesRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing ReifyReturnTypeShapes function");
  assert(function &&
         "Attempting to register an empty ReifyReturnTypeShapes function");
  reifyReturnTypeShapesRegistry[name] = function;
}

ReifyReturnTypeShapesRegistration::ReifyReturnTypeShapesRegistration(
    StringRef name, const ReifyReturnTypeShapes &function) {
  registerReifyReturnTypeShapes(name, function);
}

ReifyReturnTypeShapes mlir::reifyReturnTypeShapes(llvm::StringRef name) {
  auto &reifyReturnTypeShapesRegistry = getReifyReturnTypeShapesRegistry();
  auto it = reifyReturnTypeShapesRegistry.find(name);
  if (it != reifyReturnTypeShapesRegistry.end())
    return it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// InferBoundedReturnTypes Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<InferBoundedReturnTypes> &
getInferBoundedReturnTypesRegistry() {
  static llvm::StringMap<InferBoundedReturnTypes>
      inferBoundedReturnTypesRegistry;
  return inferBoundedReturnTypesRegistry;
}

static void
registerInferBoundedReturnTypes(StringRef name,
                                const InferBoundedReturnTypes &function) {
  auto &registry = getInferBoundedReturnTypesRegistry();
  if (registry.find(name) != registry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing InferBoundedReturnTypes function");
  assert(function &&
         "Attempting to register an empty InferBoundedReturnTypes function");
  registry[name] = function;
}

InferBoundedReturnTypesRegistration::InferBoundedReturnTypesRegistration(
    StringRef name, const InferBoundedReturnTypes &function) {
  registerInferBoundedReturnTypes(name, function);
}

InferBoundedReturnTypes mlir::inferBoundedReturnTypes(llvm::StringRef name) {
  auto &registry = getInferBoundedReturnTypesRegistry();
  auto it = registry.find(name);
  if (it != registry.end())
    return it->second;
  return nullptr;
}
