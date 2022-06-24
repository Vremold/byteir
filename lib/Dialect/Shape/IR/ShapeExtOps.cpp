//===- ShapeExtOps.cpp ----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>

using namespace mlir;
using namespace shape;
using namespace mlir::shape_ext;

#include "byteir/Dialect/Shape/ShapeExtOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ace dialect.
//===----------------------------------------------------------------------===//

void ShapeExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Shape/ShapeExtOps.cpp.inc"
      >();
}

namespace {

struct TieWithConst : public OpRewritePattern<shape_ext::TieOp> {
  using OpRewritePattern<shape_ext::TieOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape_ext::TieOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    RankedTensorType shapeType = value.getType().cast<RankedTensorType>();
    SmallVector<int64_t> shape = llvm::to_vector(shapeType.getShape());
    auto dims = op.getDims();
    SmallVector<Value> keepedDims;

    auto findNextDynamicDim = [&shape](auto it) {
      return std::find(it, shape.end(), ShapedType::kDynamicSize);
    };

    auto shpIt = findNextDynamicDim(shape.begin());
    for (auto dimIt = dims.begin(); dimIt != dims.end();
         dimIt++, shpIt = findNextDynamicDim(++shpIt)) {
      Value dimSize = *dimIt;
      Operation *defOp = dimSize.getDefiningOp();
      if (!defOp) {
        keepedDims.push_back(dimSize);
        continue;
      }

      IntegerAttr intAttr;
      if (matchPattern(dimSize, m_Constant(&intAttr))) {
        int64_t dimSizeInt = intAttr.getInt();
        *shpIt = dimSizeInt;
      } else {
        keepedDims.push_back(dimSize);
      }
    }

    if (keepedDims.size() == dims.size())
      return failure();
    value.setType(shapeType.clone(shape));
    if (keepedDims.size() == 0) {
      op->erase();
    } else {
      op.getDimsMutable().assign(keepedDims);
    }

    return success();
  }
};

} // namespace

void TieOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<TieWithConst>(context);
}

LogicalResult TieOp::verify() {
  auto rankedTensorType = getValue().getType().dyn_cast<RankedTensorType>();
  if (!rankedTensorType)
    return emitError() << "The value's type should be RankedTensorType";
  auto numDynShape =
      llvm::count_if(rankedTensorType.getShape(), [](int64_t dimSize) {
        return dimSize == ShapedType::kDynamicSize;
      });
  if (size_t(numDynShape) != getDims().size())
    return emitError() << "The number of tie's dims and the dynamic size of "
                          "the original value don't match.";

  return success();
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Shape/ShapeExtOps.cpp.inc"
