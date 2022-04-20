
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

// a mlir text loader example
static OwningOpRef<ModuleOp> parseMLIRInput(StringRef inputFilename,
                                            MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

// a mlir module traverser example
static void TraverseAllEntries(OwningOpRef<ModuleOp> &m) {
  // iterate all entry function ops in this module.
  for (FuncOp entry : m->getOps<FuncOp>()) {
    if (!entry->hasAttr(ByreDialect::getEntryPointFunctionAttrName())) {
      std::cout << "skip function " << entry.getName().str() << std::endl;
      continue;
    }
    entry.walk([&](byre::ComputeOp op) {
      std::cout << op.getCallee().str() << std::endl;
      for (auto opArg : op.getOperands()) {
        // use pointer as unique id for value
        std::cout << opArg.getAsOpaquePointer() << std::endl;
      }
    });
  }
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  // register ByteIR's dialects here
  registry.insert<mlir::byre::ByreDialect>();

  if (argc < 2) {
    llvm::errs() << "need input file arg\n";
    return 1;
  }

  MLIRContext context(registry);
  auto m = parseMLIRInput(argv[1], &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  TraverseAllEntries(m);

  return 0;
}
