//===- Passes.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TRANSFORMS_PASSES_H
#define BYTEIR_TRANSFORMS_PASSES_H

#include "byteir/Transforms/AnchoredFuncPipeline.h"
#include "byteir/Transforms/CMAE.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "byteir/Transforms/CollectFunc.h"
#include "byteir/Transforms/CondCanonicalize.h"
#include "byteir/Transforms/FuncTag.h"
#include "byteir/Transforms/GenericDeviceConfig.h"
#include "byteir/Transforms/GraphClusteringByDevice.h"
#include "byteir/Transforms/LoopTag.h"
#include "byteir/Transforms/LoopUnroll.h"
#include "byteir/Transforms/MemoryPlanning.h"
#include "byteir/Transforms/RemoveFuncBody.h"
#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "byteir/Transforms/SetArgShape.h"
#include "byteir/Transforms/SetSpace.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSES_H