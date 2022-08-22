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
class ModuleOp;

using CallMapTable = std::unordered_map<std::string, std::string>;

void populateRewriteOpToStdCallPatterns(RewritePatternSet &,
                                        const CallMapTable &);

std::unique_ptr<OperationPass<ModuleOp>>
createRewriteOpToStdCallPass(CallMapTable callTable = {});

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_REWRITEOPTOSTDCALL_H