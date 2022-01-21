//===- TilingUtils.cpp ----------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/TilingUtils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
void mlir::unpackRanges(
  ArrayRef<Range> ranges, 
  SmallVectorImpl<Value>& lbs,
  SmallVectorImpl<Value>& ubs,
  SmallVectorImpl<Value>& steps) {

  for (Range range : ranges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }
}

