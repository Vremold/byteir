//===- BatchNormTrainingFusion.cpp ----------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/IOConvertFusion.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"
#include <unordered_set>

using namespace mlir;
using namespace llvm;

namespace {

struct IOConvertFusionPattern : public RewritePattern {
  IOConvertFusionPattern(MLIRContext *context, const std::string &_opName,
                         const std::vector<int> &_inputArgIdx,
                         const std::vector<int> &_outputArgIdx,
                         const std::string &_byreComputeName)
      : RewritePattern(MatchAnyOpTypeTag(), 3, context), opName(_opName),
        inputArgIdx(_inputArgIdx.begin(), _inputArgIdx.end()),
        outputArgIdx(_outputArgIdx.begin(), _outputArgIdx.end()),
        byreComputeName(_byreComputeName) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != opName) {
      return failure();
    }
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    llvm::SmallVector<Value> inputs;
    llvm::SmallVector<Value> outputs;
    llvm::SmallVector<Operation *> ops;
    NamedAttrList attrs;
    // inputs convert
    for (size_t i = 0; i < op->getOperands().size(); i++) {
      auto value = op->getOperands()[i];
      if (inputArgIdx.find(i) == inputArgIdx.end()) {
        inputs.push_back(value);
      } else {
        auto convertOp = value.getDefiningOp<mhlo::ConvertOp>();
        if (!convertOp) {
          return failure();
        } else {
          // copy mhlo.convert
          auto _convertOp = rewriter.clone(*convertOp.getOperation());
          op->setOperand(i, _convertOp->getResult(0));
          ops.push_back(_convertOp);
          inputs.push_back(_convertOp->getOperand(0));
        }
      }
    }
    ops.push_back(op);
    // copy attrs to fusion op
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr(byreComputeName));
    for (const auto &attr : op->getAttrs()) {
      byre::appendByreComputeAttr(attrs, attr.getName().getValue(),
                                  attr.getValue());
    }
    // outputs convert
    for (size_t i = 0; i < op->getResults().size(); i++) {
      auto value = op->getResults()[i];
      if (outputArgIdx.find(i) == outputArgIdx.end()) {
        outputs.push_back(value);
      } else {
        if (UseCount(value) != 1) {
          return failure();
        }
        auto convertOp = llvm::dyn_cast_or_null<mhlo::ConvertOp>(
            (*value.getUses().begin()).getOwner());
        if (!convertOp) {
          return failure();
        } else {
          ops.push_back(convertOp.getOperation());
          outputs.push_back(convertOp.getResult());
        }
      }
    }

    auto loc = GetFusedLoc(ops, rewriter);
    llvm::SmallVector<Type> outputs_type;
    for (auto output : outputs) {
      outputs_type.push_back(output.getType());
    }
    mhlo::FusionOp fusionOp =
        rewriter.create<mhlo::FusionOp>(loc, outputs_type, inputs);
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].replaceAllUsesWith(fusionOp.getResults()[i]);
    }
    Block &block = fusionOp.fused_computation().emplaceBlock();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      for (auto _op : ops) {
        _op->moveBefore(&block, block.end());
      }

      rewriter.setInsertionPoint(&block, block.end());
      rewriter.create<mhlo::ReturnOp>(loc, outputs);
    }
    fusionOp->setAttrs(attrs.getDictionary(getContext()));

    return success();
  }
  const std::string opName;
  const std::unordered_set<int> inputArgIdx;
  const std::unordered_set<int> outputArgIdx;
  const std::string byreComputeName;
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
    LogicalResult status =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(status)) {
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