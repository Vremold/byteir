//===- HloOpt.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_HLOOPT_H
#define BYTEIR_PIPELINES_HLOOPT_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <string>


namespace mlir {

void addGenericHloFusionPatterns(OpPassManager& pm, const std::string& entry = "main");

std::unique_ptr<OperationPass<ModuleOp>>
createHloOptPipelinePass(const std::string& entry = "main", 
                         const std::string &target = "");

} // namespace mlir

#endif // BYTEIR_PIPELINES_HLOOPT_H