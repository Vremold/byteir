//===- HostOpt.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HOSTOPT_H
#define BYTEIR_PIPELINES_HOSTOPT_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;

std::unique_ptr<OperationPass<ModuleOp>>
createHostOptPipelinePass(const std::string &fileName = "host_kernels.ll");

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOSTOPT_H
