//===- CollectMhloOps.cpp --------------------------------------*--- C++
//-*-===//
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
namespace aggressive_fusion {

bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) && !llvm::isa<mhlo::CustomCallOp>(op);
}

bool isFusibleStart(Operation *) { return true; }

bool isFusibleTrigger(Operation *) { return true; }

bool isFusibleWith(Operation *, Operation *) { return true; }

bool isValidSingleOp(Operation *) { return true; }

static GenericFuserConfig config{getByteIRHloAggressiveFusionAttrName(),
                                 aggressive_fusion::isFusibleCandidate,
                                 aggressive_fusion::isFusibleStart,
                                 aggressive_fusion::isFusibleTrigger,
                                 aggressive_fusion::isFusibleWith,
                                 aggressive_fusion::isValidSingleOp};

} // namespace aggressive_fusion

// A derived fusion pass for hlo aggressive fusion, which would fuse mhlo ops
// into mhlo.fusion group as much as possible
struct HloAggressiveFusionPass
    : public GenericFusionPass<HloAggressiveFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HloAggressiveFusionPass)

  HloAggressiveFusionPass()
      : GenericFusionPass(aggressive_fusion::config, true) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("hlo-aggressive-fusion");
  }
  ::llvm::StringRef getArgument() const override {
    return "hlo-aggressive-fusion";
  }

  ::llvm::StringRef getDescription() const override {
    return "Do aggressive fusion on mhlo dialect, fuse mhlo ops into "
           "mhlo.fusion group as much as possible.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("HloAggressiveFusion");
  }
  ::llvm::StringRef getName() const override { return "HloAggressiveFusion"; }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloAggressiveFusionPass() {
  return std::make_unique<HloAggressiveFusionPass>();
}
