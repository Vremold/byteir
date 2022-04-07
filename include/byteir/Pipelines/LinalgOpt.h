//===- LinalgOpt.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_LINALGOPT_H
#define BYTEIR_PIPELINES_LINALGOPT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>

namespace mlir {

void addGenericLinalgElementwisePasses(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>>
createLinalgOptPipelinePass(const std::string &target = "");

} // namespace mlir

#endif // BYTEIR_PIPELINES_LINALGOPT_H