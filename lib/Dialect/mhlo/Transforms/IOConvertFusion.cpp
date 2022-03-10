//===- BatchNormTrainingFusion.cpp ----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/IOConvertFusion.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Dialect/mhlo/Transforms/FusionUtil.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

// Note IOConvert will keep input/output sequence order as orginal op
struct IOConvertFusionPattern : public RewritePattern {
  IOConvertFusionPattern(
    MLIRContext *context, StringRef _opName,
    ArrayRef<int> _inputArgIdx,
    ArrayRef<int> _outputArgIdx, 
    StringRef _byreComputeName)
      : RewritePattern(MatchAnyOpTypeTag(), 3, context), opName(_opName),
        inputArgIdx(_inputArgIdx.begin(), _inputArgIdx.end()),
        outputArgIdx(_outputArgIdx.begin(), _outputArgIdx.end()),
        byreComputeName(_byreComputeName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    
    // early termination
    if (op->getName().getStringRef() != opName || 
        op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    MhloFusionPattern pattern;
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;

    // handle input's convert
    for (unsigned idx = 0; idx < op->getNumOperands(); ++idx) {
      auto value = op->getOperand(idx);
      auto defOp = value.getDefiningOp();
      if (isa_and_nonnull<mhlo::ConvertOp>(defOp)) {
        auto cloned = ReplicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
        inputs.push_back(cloned->getOperand(0));
      } else if (IsSplatMhloConstant(defOp)) {
        auto cloned = ReplicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
      } else {
        inputs.push_back(value);
      }
    }

    // op itself
    pattern.push_back(op);

    // handle output's convert
    for (unsigned idx = 0; idx < op->getNumResults(); ++idx) {
      auto value = op->getResult(idx);

      if (UseCount(value) == 0) {
        continue;
      }

      if (UseCount(value) > 1) {
        outputs.push_back(value);
        continue;
      }

      auto user = *value.getUsers().begin();
      if (isa_and_nonnull<mhlo::ConvertOp>(user)) {
        pattern.push_back(user);
        outputs.push_back(user->getResult(0));
      } else {
        outputs.push_back(value);
      }
    }

    // terminate if only single op
    if (pattern.size() == 1) return failure();

    NamedAttrList attrs;
    // copy attrs to fusion op
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr(byreComputeName));
    for (const auto &attr : op->getAttrs()) {
      byre::appendByreComputeAttr(attrs, attr.getName().getValue(),
                                  attr.getValue());
    }

    mhlo::FusionOp fusionOp = creatMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }

  StringRef opName;
  const SmallDenseSet<int> inputArgIdx;
  const SmallDenseSet<int> outputArgIdx;
  StringRef byreComputeName;
};

struct IOConvertFusionPass : public IOConvertFusionBase<IOConvertFusionPass> {
  IOConvertFusionPass() = default;
  IOConvertFusionPass(std::string _opName, std::vector<int> _inputArgIdx,
                      std::vector<int> _outputArgIdx,
                      std::string _byreComputeName)
      : _opName(_opName), _inputArgIdx(_inputArgIdx),
        _outputArgIdx(_outputArgIdx), _byreComputeName(_byreComputeName) {
    this->opName = "";
    this->inputArgIdx = {};
    this->outputArgIdx = {};
    this->byreComputeName = "";
  }

  void runOnOperation() override {
    if (this->opName != "") {
      this->_opName = this->opName;
      this->_byreComputeName = this->byreComputeName;
      for (auto &idx : this->inputArgIdx) {
        _inputArgIdx.push_back(idx);
      }
      for (auto &idx : this->outputArgIdx) {
        _outputArgIdx.push_back(idx);
      }
    }

    if (_opName == "" || _byreComputeName == "") {
      signalPassFailure();
    }
    if (_inputArgIdx.size() == 0 && _outputArgIdx.size() == 0) {
      signalPassFailure();
    }

    FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<IOConvertFusionPattern>(context, _opName, _inputArgIdx,
                                         _outputArgIdx, _byreComputeName);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError("IOConvertFusionPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }

  std::string _opName;
  std::vector<int> _inputArgIdx;
  std::vector<int> _outputArgIdx;
  std::string _byreComputeName;
};
} // namespace

void populateIOConvertBatchNormPattern(RewritePatternSet &patterns) {
  patterns.add(std::make_unique<IOConvertFusionPattern>(
      patterns.getContext(), "mhlo.batch_norm_training", std::vector<int>{0},
      std::vector<int>{0}, "BatchNormTrainingOp"));
  patterns.add(std::make_unique<IOConvertFusionPattern>(
      patterns.getContext(), "mhlo.batch_norm_grad", std::vector<int>{0, 4},
      std::vector<int>{0}, "BatchNormGradOp"));
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createIOConvertFusionPass() {
  return std::make_unique<IOConvertFusionPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createIOConvertFusionPass(
    std::string opName, std::vector<int> inputArgIdx,
    std::vector<int> outputArgIdx, std::string byreComputeName) {
  return std::make_unique<IOConvertFusionPass>(opName, inputArgIdx,
                                               outputArgIdx, byreComputeName);
}