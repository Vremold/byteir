//===- Common.cpp -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Pipelines/Common.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::addCleanUpPassPipeline(OpPassManager& pm) {
  pm.addPass(createCSEPass());
  pm.addPass(createSCCPPass());
  pm.addPass(createCanonicalizerPass());
}

void mlir::addMultiCSEPipeline(OpPassManager& pm, unsigned cnt) {
  for (unsigned i = 0; i < cnt; ++i) {
    pm.addPass(createCSEPass());
  }

}

