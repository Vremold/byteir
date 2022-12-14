//===- FuseElementwise.h --------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_FUSEELEMENTWISE_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_FUSEELEMENTWISE_H

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class DominanceInfo;
class PostDominanceInfo;

namespace linalg {

// TODO: maybe move the following to transform.h
bool isProducerElementwiseOpFusable(OpOperand *consumerOpOperand);

void populateElementwiseOpsProducerConsumerFusionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlElementwiseOpFusion,
    DominanceInfo &dom, PostDominanceInfo &post);

} // namespace linalg

namespace linalg_ext {
void populateInsertLinalgExtAliasForSharedInputFusionPatterns(
    RewritePatternSet &patterns, DominanceInfo &dom);

void populateRemoveLinalgExtAliasPattern(RewritePatternSet &patterns);
} // namespace linalg_ext

std::unique_ptr<Pass>
createLinalgElementwiseFusionExtPass(bool enableSharedInput = false);

std::unique_ptr<Pass> createLinalgElementwiseFusionExtPass(
    const linalg::ControlFusionFn &controlElementwiseOpFusion,
    bool enableSharedInput = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_FUSEELEMENTWISE_H
