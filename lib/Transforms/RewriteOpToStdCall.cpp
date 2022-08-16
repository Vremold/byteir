//===- ConvertOpToStdCall.cpp ---------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

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
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  funcOp.setPrivate();
  return fnNameAttr;
}

struct RewriteOpToStdCallPattern : public RewritePattern {
  RewriteOpToStdCallPattern(MLIRContext *context, const CallMapTable &lut)
      : RewritePattern(MatchAnyOpTypeTag(), 3, context), callMapTable(lut) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    std::string opName = op->getName().getStringRef().str();
    auto iter = callMapTable.find(opName);
    if (iter != callMapTable.end()) {
      FlatSymbolRefAttr libraryCallName =
          getLibraryCallSymbolRef(op, rewriter, iter->second);
      rewriter.replaceOpWithNewOp<func::CallOp>(op, libraryCallName.getValue(),
                                                TypeRange(), op->getOperands());
      return success();
    }
    return failure();
  }
  const CallMapTable &callMapTable;
};

struct RewriteOpToStdCallPass
    : public RewriteOpToStdCallBase<RewriteOpToStdCallPass> {
  RewriteOpToStdCallPass() = default;
  RewriteOpToStdCallPass(CallMapTable lut) : callMapTable(lut) {
    this->callTable = {};
  }
  void runOnOperation() override {

    // parse callTable into callMapTable
    if (this->callTable.size() != 0) {
      for (auto &table : this->callTable) {
        int semicolon = table.find(':');
        this->callMapTable[table.substr(0, semicolon)] =
            table.substr(semicolon + 1);
      }
    }

    if (this->callMapTable.size() == 0) {
      return signalPassFailure();
    }

    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteOpToStdCallPattern>(patterns.getContext(),
                                            this->callMapTable);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
  CallMapTable callMapTable;
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRewriteOpToStdCallPass(CallMapTable callTable) {
  return std::make_unique<RewriteOpToStdCallPass>(callTable);
}
