//===- TotalBufferize.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/TotalBufferize.h"
#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "byteir/Dialect/Ace/Passes.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createByteIRTotalBufferizePipeline(OpPassManager &pm) {
  pm.addPass(createConvertHloToLHloPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createAceBufferizePass());
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  addCleanUpPassPipeline(pm);
  // clean-up possible redudant copy-removal from bufferization
  // TODO: enable it after fixing crash
  // pm.addNestedPass<func::FuncOp>(createCopyRemovalPass());
}
