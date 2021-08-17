//===- RewriteOpToStdCall.h -----------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_REWRITEOPTOSTDCALL_H
#define BYTEIR_TRANSFORMS_REWRITEOPTOSTDCALL_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir {

using CallTable = std::unordered_map<std::string, std::string>;

std::unique_ptr<OperationPass<ModuleOp>>
createRewriteOpToStdCallPass(CallTable callTable = {});

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_REWRITEOPTOSTDCALL_H