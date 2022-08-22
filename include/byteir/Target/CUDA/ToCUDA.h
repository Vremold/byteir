//===- ToCUDA.h -----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TARGET_CUDA_TOCUDA_H
#define BYTEIR_TARGET_CUDA_TOCUDA_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace byteir {

void registerToCUDATranslation();

/// Translates the given operation to CUDA code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
mlir::LogicalResult translateToCUDA(mlir::Operation *op, llvm::raw_ostream &os,
                                    bool declareVariablesAtTop = false,
                                    bool kernelOnly = false,
                                    bool externC = false);
} // namespace byteir

#endif // BYTEIR_TARGET_CUDA_TOCUDA_H
