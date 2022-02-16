//===- ConvertOpToStdCall.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "./PassDetail.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static FlatSymbolRefAttr
getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter,
                        const std::string &calleeName) {
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), calleeName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr())) {
    return fnNameAttr;
  }

  assert(op->getNumResults() == 0 &&
         "std call for operation can be generated only for ops that "
         "have void return types");
  auto libFnType = rewriter.getFunctionType(op->getOperandTypes(), {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  FuncOp funcOp =
      rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType);
  funcOp.setPrivate();
  return fnNameAttr;
}

struct RewriteOpToStdCallPattern : public RewritePattern {
  RewriteOpToStdCallPattern(MLIRContext *context, const CallTable &_callTable)
      : RewritePattern(MatchAnyOpTypeTag(), 3, context), callTable(_callTable) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    std::string opName = op->getName().getStringRef().str();
    auto iter = callTable.find(opName);
    if (iter != callTable.end()) {
      FlatSymbolRefAttr libraryCallName =
          getLibraryCallSymbolRef(op, rewriter, iter->second);
      rewriter.replaceOpWithNewOp<mlir::CallOp>(op, libraryCallName.getValue(),
                                                TypeRange(), op->getOperands());
      return success();
    }
    return failure();
  }
  const CallTable &callTable;
};

struct RewriteOpToStdCallPass
    : public RewriteOpToStdCallBase<RewriteOpToStdCallPass> {
  RewriteOpToStdCallPass() = default;
  RewriteOpToStdCallPass(CallTable _callTable) : _callTable(_callTable) {
    this->callTable = {};
  }
  void runOnOperation() override {
    if (this->callTable.size() != 0) {
      for (auto &table : this->callTable) {
        int semicolon = table.find(':');
        this->_callTable[table.substr(0, semicolon)] =
            table.substr(semicolon + 1);
      }
    }

    if (this->_callTable.size() == 0) {
      signalPassFailure();
    }

    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect, memref::MemRefDialect>();
    target.addLegalOp<ModuleOp, FuncOp, ReturnOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteOpToStdCallPattern>(patterns.getContext(),
                                            this->_callTable);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
  CallTable _callTable;
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRewriteOpToStdCallPass(CallTable callTable) {
  return std::make_unique<RewriteOpToStdCallPass>(callTable);
}
