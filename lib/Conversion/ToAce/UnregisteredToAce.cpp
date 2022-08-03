//===- UnregisteredToAce.cpp ----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToAce/UnregisteredToAce.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include "../PassDetail.h"

using namespace mlir;

namespace {

static void WrapUnregisteredOpWithOpaque(Operation *op) {
  Block *block = op->getBlock();
  OpBuilder opBuilder(block, block->begin());
  opBuilder.setInsertionPoint(op);
  ace::OpaqueOp opaqueOp = opBuilder.create<ace::OpaqueOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands());
  Block *innerBlock = opaqueOp.addEntryBlock();
  op->moveBefore(innerBlock, innerBlock->end());
  op->setOperands(innerBlock->getArguments());
  opBuilder.setInsertionPoint(innerBlock, innerBlock->end());
  opBuilder.create<ace::ReturnOp>(op->getLoc(), op->getResults());

  for (auto outputAndResult :
       llvm::zip(op->getResults(), opaqueOp.getResults())) {
    Value output = std::get<0>(outputAndResult);
    Value opaqueResult = std::get<1>(outputAndResult);
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      if (use.getOwner()->getBlock() != innerBlock)
        use.set(opaqueResult);
    }
  }
}

struct ConvertUnregisteredToAcePass
    : public ConvertUnregisteredToAceBase<ConvertUnregisteredToAcePass> {

  ConvertUnregisteredToAcePass() : ConvertUnregisteredToAceBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](mlir::Operation *op) {
      if (!op->getDialect()) {
        WrapUnregisteredOpWithOpaque(op);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertUnregisteredToAcePass() {
  return std::make_unique<ConvertUnregisteredToAcePass>();
}
