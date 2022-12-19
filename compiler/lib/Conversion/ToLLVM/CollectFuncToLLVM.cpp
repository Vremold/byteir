//===- CollectFuncToLLVM.cpp -------------------------------------- C++ -*-===//
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

#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Byre/Common.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "../PassDetail.h"

using namespace mlir;

static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";

namespace {
ModuleOp getOrCreateLLVMSubmodule(ModuleOp m) {
  for (auto &op : m.getBody()->without_terminator()) {
    if (auto sm = dyn_cast<ModuleOp>(op)) {
      if (sm->hasAttr(getByteIRLLVMModuleAttrName())) {
        return sm;
      }
    }
  }

  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  ModuleOp sm = builder.create<ModuleOp>(m.getLoc());
  sm->setAttr(getByteIRLLVMModuleAttrName(), UnitAttr::get(m->getContext()));
  return sm;
}

LogicalResult processSingleFunc(func::FuncOp func, ModuleOp sm) {
  FunctionType funcType = func.getFunctionType();
  OpBuilder b(sm.getRegion());
  func::FuncOp newFunc =
      b.create<func::FuncOp>(func.getLoc(), func.getName(), funcType);
  BlockAndValueMapping bvm;
  func.getRegion().cloneInto(&newFunc.getRegion(), bvm);
  // TODO: pass llvm config attributes to new function
  // TODO: make c interface optional
  newFunc->setAttr(kEmitIfaceAttrName, UnitAttr::get(newFunc->getContext()));
  // TODO: handle alias op in function body
  func.eraseBody();
  func.setPrivate();
  return success();
}

struct CollectFuncToLLVMPass
    : public CollectFuncToLLVMBase<CollectFuncToLLVMPass> {

  CollectFuncToLLVMPass() : CollectFuncToLLVMBase() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<func::FuncOp> funcs;
    for (auto func : m.getOps<func::FuncOp>()) {
      if (auto nameAttr =
              func->getAttrOfType<StringAttr>(byre::getByreComputeName())) {
        if (nameAttr.getValue() == getByteIRLLVMJITOpKernelName()) {
          funcs.push_back(func);
        }
      }
    }
    if (funcs.empty()) {
      return;
    }
    ModuleOp sm = getOrCreateLLVMSubmodule(m);
    for (auto func : funcs) {
      if (failed(processSingleFunc(func, sm))) {
        signalPassFailure();
      }
    }
    OpPassManager pm(sm.getOperationName());
    pm.addPass(bufferization::createBufferResultsToOutParamsPass());
    pm.addNestedPass<func::FuncOp>(
        bufferization::createPromoteBuffersToStackPass());
    if (mlir::failed(runPipeline(pm, sm))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createCollectFuncToLLVMPass() {
  return std::make_unique<CollectFuncToLLVMPass>();
}
