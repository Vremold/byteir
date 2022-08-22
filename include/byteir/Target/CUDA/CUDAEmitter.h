//===- CUDAEmitter.h ------------------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

// Some code are CppEmitter.h and TranslateToCpp.cpp of LLVM
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TARGET_CUDA_CUDAEMITTER_H
#define BYTEIR_TARGET_CUDA_CUDAEMITTER_H

#include "byteir/Target/Cpp/CppEmitter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace byteir {

class CUDAEmitter : public CppEmitter {
public:
  explicit CUDAEmitter(llvm::raw_ostream &os, bool declareVariablesAtTop,
                       bool kernelOnly, bool externC);

  virtual mlir::LogicalResult emitOperation(mlir::Operation &op,
                                            bool trailingSemicolon) override;

  bool shouldEmitKernelOnly() { return kernelOnly; };

  bool shouldEmitExternC() { return externC; };

protected:
  // emit kernel only
  bool kernelOnly;

  // add extern C
  bool externC;
};

} // namespace byteir

#endif // BYTEIR_TARGET_CUDA_CUDAEMITTER_H
