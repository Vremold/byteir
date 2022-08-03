//===- HloToLHlo.h --------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_CONVERSION_HLOTOLHLO_H
#define BYTEIR_CONVERSION_HLOTOLHLO_H

#include "mlir/Pass/Pass.h"
#include <memory>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;

// Collection of rewrite patterns for lowering of HLO to LHLO dialect.
void populateHLOToLHLOConversionPatternExtension(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns);

std::unique_ptr<OperationPass<ModuleOp>> createConvertHloToLHloPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_HLOTOLHLO_H