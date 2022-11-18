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

def LinalgDataPlace : Pass<"linalg-data-place", "func::FuncOp"> {
  let summary = "Data-Place Linalg's buffer to a local buffer with space";
  let constructor = "mlir::createLinalgDataPlacePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "arith::ArithDialect",
  ];
  let options = [
    ListOption<"memSpaces", "mem-spaces", "int64_t",
               "data placement memory spaces",
               "llvm::cl::ZeroOrMore">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgPrefetch
//===----------------------------------------------------------------------===//

def LinalgPrefetch : Pass<"linalg-prefetch", "func::FuncOp"> {
  let summary = "Prefetch Linalg's copies within loop";
  let constructor = "mlir::createLinalgPrefetchPass()";
  let options = [
    Option<"prefetchCnt", "count", "int64_t", /*default=*/"1",
           "prefetch count, <= 0 implies disabled">,
    Option<"unroll", "unroll", "bool", /*default=*/"false",
           "unroll prefetch">,	
  ];
}

//===----------------------------------------------------------------------===//
// LinalgScopeTiling
//===----------------------------------------------------------------------===//

def LinalgScopeTiling : Pass<"linalg-scope-tile", "func::FuncOp"> {
  let summary = "Perform Tile linalg operations within a scope";
  let constructor = "mlir::createLinalgScopeTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"tileAxis", "axis", "int64_t", /*default=*/"0",
           "tile axis">,
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"0",
           "tile size">,
    Option<"parallelizeReduction", "par-reduce", "bool", /*default=*/"false",
            "force parallelize reduction">,	
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"for\"",
           "Specify the type of loops to generate: scf, affine, tiled_loop">,
    Option<"distributionType", "distribution-type", "std::string", /*default=*/"",
           "DistributionTypes (if loop-type=tiled_loop)">,
  ];
}

#endif // BYTEIR_LINALG_PASSES