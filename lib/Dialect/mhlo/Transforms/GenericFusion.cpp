//===- GenericFusion.cpp --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/mhlo/Transforms/GenericFusionCommon.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {
namespace elementwise {

bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
          isMhloConstantLike(op) ||
          isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(op));
}

bool isFusibleStart(Operation *op) { return true; }

bool isFusibleTrigger(Operation *op) {
  if (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
      op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
      isa<mhlo::ReshapeOp>(op)) {
    return true;
  }

  if (isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp>(op)) {
    auto val = op->getOperand(0);
    auto def_op = val.getDefiningOp();
    return def_op && isMhloConstantLike(def_op);
  }

  return false;
}

bool isFusibleWith(Operation *target, Operation * /*start*/) {
  return target->hasTrait<::mlir::OpTrait::Elementwise>() ||
         target->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
         isMhloConstantLike(target) ||
         isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(
             target);
}

bool isValidSingleOp(Operation *op) {
  return op->hasTrait<::mlir::OpTrait::Elementwise>() ||
         op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>();
}

static GenericFuserConfig config{
    getByteIRElementwiseFusionAttrName(), elementwise::isFusibleCandidate,
    elementwise::isFusibleStart,          elementwise::isFusibleTrigger,
    elementwise::isFusibleWith,           elementwise::isValidSingleOp};

} // namespace elementwise

namespace matmul_epilogue {

bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          op->hasTrait<mhlo::OpTrait::BroadcastingElementwise>() ||
          isMhloConstantLike(op) ||
          isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp,
              mhlo::DotOp>(op));
}

bool isFusibleStart(Operation *op) { return isa<mhlo::DotOp>(op); }

bool isFusibleTrigger(Operation *op) {
  // trigger fuse for anything but dot
  return !isa<mhlo::DotOp>(op);
}

bool isFusibleWith(Operation * /*target*/, Operation * /*start*/) {
  return true;
}

bool isValidSingleOp(Operation *op) { return false; }

static GenericFuserConfig config{getByteIRMatmulEpilogueFusionAttrName(),
                                 matmul_epilogue::isFusibleCandidate,
                                 matmul_epilogue::isFusibleStart,
                                 matmul_epilogue::isFusibleTrigger,
                                 matmul_epilogue::isFusibleWith,
                                 matmul_epilogue::isValidSingleOp};

} // namespace matmul_epilogue

// a derived fusion pass for elementwise
struct ElementwiseFusionPass : public GenericFusionPass<ElementwiseFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementwiseFusionPass)

  ElementwiseFusionPass(bool clusterSingleOp)
      : GenericFusionPass(elementwise::config, clusterSingleOp) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-element");
  }
  ::llvm::StringRef getArgument() const override { return "fuse-element"; }

  ::llvm::StringRef getDescription() const override {
    return "Fuse elementwise op";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ElementFusion");
  }
  ::llvm::StringRef getName() const override { return "ElementFusion"; }
};

// a derived fusion pass for matmul epilogue fusion
struct MatmulEpilogueFusionPass
    : public GenericFusionPass<MatmulEpilogueFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulEpilogueFusionPass)

  MatmulEpilogueFusionPass()
      : GenericFusionPass(matmul_epilogue::config, false) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-matmul-epilogue");
  }
  ::llvm::StringRef getArgument() const override {
    return "fuse-matmul-epilogue";
  }

  ::llvm::StringRef getDescription() const override {
    return "Fuse Matmul with elementwise epilogue op";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("MatmulEpilogueFusion");
  }
  ::llvm::StringRef getName() const override { return "MatmulEpilogueFusion"; }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createElementFusionPass(bool clusterSingleElemwiseOp) {
  return std::make_unique<ElementwiseFusionPass>(clusterSingleElemwiseOp);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createMatmulEpilogueFusionPass() {
  return std::make_unique<MatmulEpilogueFusionPass>();
}
