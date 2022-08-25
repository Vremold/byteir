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

#include "byteir/Conversion/Passes.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/MemRef/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/Shape/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/GPU/Passes.h"
#include "byteir/Pipelines/Host/Passes.h"
#include "byteir/Pipelines/Passes.h"
#include "byteir/Transforms/Passes.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

namespace byteir {
namespace test {
void registerTestConvertFuncToCustomCallPass();
void registerTestConvertInsertionPass();
void registerTestFuncArgRearrangementPass();
void registerTestPrintArgSideEffectPass();
void registerTestPrintLivenessPass();
void registerTestPrintUseRangePass();
void registerTestPrintSymbolicShapePass();
void registerTestPrintShapeAnalysisPass();
void registerTestByreOpInterfacePass();
} // namespace test
} // namespace byteir

#ifdef BYTEIR_INCLUDE_TESTS
void registerTestPasses() {
  byteir::test::registerTestConvertFuncToCustomCallPass();
  byteir::test::registerTestConvertInsertionPass();
  byteir::test::registerTestFuncArgRearrangementPass();
  byteir::test::registerTestPrintArgSideEffectPass();
  byteir::test::registerTestPrintLivenessPass();
  byteir::test::registerTestPrintUseRangePass();
  byteir::test::registerTestPrintSymbolicShapePass();
  byteir::test::registerTestPrintShapeAnalysisPass();
  byteir::test::registerTestByreOpInterfacePass();
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::hlo::registerAllHloPasses();

  registerByteIRConversionPasses();
  registerByteIRTransformsPasses();
  registerByteIRAffinePasses();
  registerByteIRByrePasses();
  registerByteIRLinalgPasses();
  registerByteIRMemRefPasses();
  registerByteIRMhloPassesExt();
  registerByteIRSCFPasses();
  registerByteIRShapePasses();

  registerByteIRPipelinesPasses();
  registerByteIRGPUPipelinesPasses();
  registerByteIRHostPipelinesPasses();

#ifdef BYTEIR_INCLUDE_TESTS
  registerTestPasses();
#endif

  DialectRegistry registry;
  registerAllDialects(registry);

  // register ByteIR's dialects here
  registry.insert<mlir::ace::AceDialect>();
  registry.insert<mlir::byre::ByreDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::lace::LaceDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::shape_ext::ShapeExtDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ByteIR pass driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
