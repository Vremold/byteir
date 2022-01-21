//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
//
// Copyright (c) ByteDance Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0
//
//===----------------------------------------------------------------------===//


#ifndef BYTEIR_LINALG_PASSES
#define BYTEIR_LINALG_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LinalgDataPlace
//===----------------------------------------------------------------------===//

def LinalgDataPlace : Pass<"linalg-data-place", "mlir::FuncOp"> {
  let summary = "Data-Place Linalg's buffer to a local buffer with space";
  let constructor = "mlir::createLinalgDataPlacePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "arith::ArithmeticDialect",
  ];
  let options = [
    ListOption<"memSpaces", "mem-spaces", "int64_t",
               "data placement memory spaces",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgScopeTiling
//===----------------------------------------------------------------------===//

def LinalgScopeTiling : Pass<"linalg-scope-tile", "mlir::FuncOp"> {
  let summary = "Perform Tile linalg operations within a scope";
  let constructor = "mlir::createLinalgScopeTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "arith::ArithmeticDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string", /*default=*/"",
           "use anchored op's iteration space for tiling">,
    Option<"tileAxis", "axis", "int64_t", /*default=*/"0",
           "tile axis">,
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"0",
           "tile size">,
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"for\"",
           "Specify the type of loops to generate: for, parallel or "
           "affine, tiled_loop">,
    Option<"distributionType", "distribution-type", "std::string", /*default=*/"",
           "DistributionTypes (if loop-type=tiled_loop)">
  ];
}

#endif // BYTEIR_LINALG_PASSES