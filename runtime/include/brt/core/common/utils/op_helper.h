//===- op_helper.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "byteir/Dialect/Byre/ByreDialect.h"

// TODO move this file to generic provider

namespace brt {

bool IsLocalAlias(mlir::Operation *op);

bool IsArgAlias(mlir::Operation *op);

bool IsAliasOp(mlir::Operation *op);

size_t GetAliasOffsetInByte(mlir::Operation *op);

bool IsAllocOp(mlir::Operation *op);

// return whether op is dynamic allocation, corresponding dynamic sizes will be
// set if true
bool IsDynamicAllocOp(mlir::Operation *op,
                      std::vector<mlir::Value> &dynamicSizes);

bool IsShapeComputeOp(mlir::Operation *op);

} // namespace brt
