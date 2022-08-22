//===- InsertTieShape.cpp ------------------------------------------ C++ --===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/Transforms/InsertTieShape.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

struct InsertTieShapePass : public InsertTieShapeBase<InsertTieShapePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp);
    funcOp.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        builder.setInsertionPointAfter(op);
        if (auto shape = result.getType().dyn_cast<RankedTensorType>()) {
          if (!shape.hasStaticShape()) {
            SmallVector<Value> dims;
            for (int64_t i = 0; i < shape.getRank(); ++i) {
              if (shape.isDynamicDim(i)) {
                dims.push_back(
                    builder.create<tensor::DimOp>(op->getLoc(), result, i));
              }
            }
            if (dims.size() > 0)
              builder.create<shape_ext::TieOp>(op->getLoc(), result, dims);
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createInsertTieShapePass() {
  return std::make_unique<InsertTieShapePass>();
}
