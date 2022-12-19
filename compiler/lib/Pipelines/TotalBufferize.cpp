//===- TotalBufferize.cpp -------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
  invokeOpPassPipelineBuilder(
      [](OpPassManager &pm) {
        pm.addPass(createConvertHloToLHloPass());
        pm.addPass(createCSEPass());
        pm.addNestedPass<func::FuncOp>(createAceBufferizePass());
        pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
        pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
        addCleanUpPassPipeline(pm);
        // clean-up possible redudant copy-removal from bufferization
        // TODO: enable it after fixing crash
        // pm.addNestedPass<func::FuncOp>(createCopyRemovalPass());
      },
      pm);
}
