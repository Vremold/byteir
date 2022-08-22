//===- TranslateRegistration.cpp ------------------------------*--- C++ -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//

// Some code from TranslateRegistration.cpp of LLVM
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Target/CUDA/ToCUDA.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// CUDA registration
//===----------------------------------------------------------------------===//

// some code from mlir's registerToCppTranslation
void byteir::registerToCUDATranslation() {
  static llvm::cl::OptionCategory CudaCat("CUDA-Emitter",
                                          "CUDA-Emitter options");

  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-var-at-top-cuda",
      llvm::cl::desc("Declare variables at top when emitting CUDA"),
      llvm::cl::init(false), llvm::cl::cat(CudaCat));

  static llvm::cl::opt<bool> forceExternC(
      "force-extern-c", llvm::cl::desc("Add Extern C in a kernel"),
      llvm::cl::init(false), llvm::cl::cat(CudaCat));

  static llvm::cl::opt<bool> kernelOnly(
      "kernel-only", llvm::cl::desc("Emit kernel only"), llvm::cl::init(false),
      llvm::cl::cat(CudaCat));

  TranslateFromMLIRRegistration reg(
      "emit-cuda",
      [](ModuleOp module, raw_ostream &output) {
        return byteir::translateToCUDA(module, output, declareVariablesAtTop,
                                       kernelOnly, forceExternC);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithmeticDialect,
                        cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect,
                        gpu::GPUDialect,
                        memref::MemRefDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}
