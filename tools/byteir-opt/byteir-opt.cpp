//===- byteir-opt.cpp - ByteIR's MLIR Optimizer Driver---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for byteir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright (c) ByteDance.

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Conversion/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "mlir-hlo/Transforms/register_passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"


using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::hlo::registerAllHloPasses();

  registerByteIRConversionPasses();
  registerByteIRTransformsPasses();
  registerByteIRAffinePasses();
  registerByteIRMhloPasses();
  DialectRegistry registry;
  registerAllDialects(registry);

  // register ByteIR's dialects here
  registry.insert<mlir::ace::AceDialect>();
  registry.insert<mlir::byre::ByreDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ByteIR pass driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
