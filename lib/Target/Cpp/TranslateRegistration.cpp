//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Target/Cpp/ToCpp.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

// some code from mlir's registerToCppTranslation
void byteir::registerToCppTranslation() {
  static llvm::cl::OptionCategory CppCat("Cpp-Emitter",
    "Cpp-Emitter options");

  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-var-at-top-cpp",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false),
      llvm::cl::cat(CppCat));

  TranslateFromMLIRRegistration reg(
      "emit-cpp",
      [](ModuleOp module, raw_ostream &output) {
        return byteir::translateToCpp(
            module, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithmeticDialect,
                        emitc::EmitCDialect,
                        memref::MemRefDialect,
                        StandardOpsDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}
