//===- InitAllStats.h -----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_STAT_INITALLSTATS_H
#define MLIR_STAT_INITALLSTATS_H

#include "byteir/Stat/AllocCnt/AllocCnt.h"
#include "byteir/Stat/OpCnt/OpCnt.h"

namespace byteir {
inline void registerAllStatistics() {
  registerAllocCntStatistics();
  registerOpCntStatistics();
}
} // namespace byteir

#endif // MLIR_STAT_INITALLSTATS_H