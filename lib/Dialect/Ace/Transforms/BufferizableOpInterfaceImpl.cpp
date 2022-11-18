//===- BufferizableOpInterfaceImpl.cpp ------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Ace/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Lace/LaceDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
struct ConstOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstOpInterface,
                                                    ace::ConstOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto constOp = cast<ace::ConstOp>(op);

    // Allocate outputs.
    auto output = constOp.getOutput();
    auto tensorType = output.getType().cast<RankedTensorType>();
    if (!tensorType)
      return failure();
    // don't dealloc constant, alawys mark as escaped
    FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
        rewriter, op->getLoc(), output,
        /* escape */ true, options);
    if (failed(tensorAlloc))
      return failure();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value outputBuffer = rewriter.create<bufferization::ToMemrefOp>(
        op->getLoc(), memrefType, *tensorAlloc);

    rewriter.create<lace::ConstOp>(op->getLoc(), constOp.getValue(),
                                   outputBuffer);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, outputBuffer);
    return success();
  }
};

struct CustomCallOpInterface
    : public BufferizableOpInterface::ExternalModel<CustomCallOpInterface,
                                                    ace::CustomCallOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false; // Arguments are read-only.
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *, OpOperand &,
                                            const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto customCallOp = cast<ace::CustomCallOp>(op);

    // Bufferize arguments.
    SmallVector<Value> bufferArgs;
    for (OpOperand &operand : customCallOp->getOpOperands()) {
      if (!operand.get().getType().isa<TensorType>())
        return failure();
      FailureOr<Value> operandBuffer =
          getBuffer(rewriter, operand.get(), options);
      if (failed(operandBuffer))
        return failure();
      bufferArgs.push_back(*operandBuffer);
    }

    // Allocate outputs.
    for (OpResult result : customCallOp->getOpResults()) {
      auto tensorType = result.getType().cast<RankedTensorType>();
      if (!tensorType)
        return failure();
      AnalysisState analysisState(options);
      FailureOr<Value> tensorAlloc =
          bufferization::allocateTensorForShapedValue(
              rewriter, op->getLoc(), result,
              analysisState.isTensorYielded(result), options);
      if (failed(tensorAlloc))
        return failure();
      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      Value resultBuffer = rewriter.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, *tensorAlloc);
      bufferArgs.push_back(resultBuffer);
    }

    auto laceOp = rewriter.create<lace::CustomCallOp>(
        op->getLoc(), llvm::None, bufferArgs, op->getAttrs());
    laceOp->setAttr(laceOp.getOperandSegmentSizeAttr(),
                    rewriter.getDenseI32ArrayAttr(
                        {static_cast<int32_t>(op->getNumOperands()),
                         static_cast<int32_t>(op->getNumResults())}));
    bufferization::replaceOpWithBufferizedValues(
        rewriter, op, makeArrayRef(bufferArgs).slice(op->getNumOperands()));
    return success();
  }
};
} // namespace

void mlir::ace::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ace::AceDialect *dialect) {
    ace::ConstOp::attachInterface<ConstOpInterface>(*ctx);
    ace::CustomCallOp::attachInterface<CustomCallOpInterface>(*ctx);
  });
}
