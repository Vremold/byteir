//===- ToPTX.h -----------------------------------------------------===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TARGET_PTX_TOPTX_H
#define BYTEIR_TARGET_PTX_TOPTX_H

#include "byteir/Target/Common/Common.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>
#include <string>

namespace mlir {

void registerToPTXTranslation();

LogicalResult translateToPTX(Operation *op, raw_ostream &os,
                             const std::string &prefix = "out",
                             OptLevel level = OptLevel::O3,
                             const std::string &gpuArch = "sm_70",
                             bool dumpPtx = false, bool saveTemp = false,
                             bool verbose = false);

} // namespace mlir

#endif // BYTEIR_TARGET_PTX_TOPTX_H