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
// LinalgScopeTiling
//===----------------------------------------------------------------------===//

def LinalgScopeTiling : FunctionPass<"linalg-scope-tile"> {
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