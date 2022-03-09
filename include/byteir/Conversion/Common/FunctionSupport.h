//===- FunctionSupport.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_COMMON_H
#define BYTEIR_CONVERSION_COMMON_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mlir {

// 
// LWC NOTE This implementation DO NOT support inout,
// meaning directly returning an input as an results
// LWC NOTE Also DO NOT support duplicated results.
//
void replicateFuncOpResults(mlir::FuncOp funcOp);

void replicateFuncOpResults(mlir::FuncOp funcOp,
                           std::function<void(mlir::ReturnOp)> retOpHandling);

void relocateFuncOpConstantLike(
    mlir::FuncOp funcOp, 
    std::function<bool(mlir::Operation*)> checkOp,
    std::function<std::tuple<mlir::Value, NamedAttrList>(mlir::Operation *)>
        getValue);

} // namespace mlir

#endif // BYTEIR_CONVERSION_COMMON_H