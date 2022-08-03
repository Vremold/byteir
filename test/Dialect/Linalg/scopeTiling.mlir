// RUN: byteir-opt %s -linalg-scope-tile="axis=0 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS0
// RUN: byteir-opt %s -linalg-scope-tile="axis=1 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS1
// RUN: byteir-opt %s -linalg-scope-tile="axis=2 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS2

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_element_static(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %0 = memref.alloc() : memref<128x64xf32>
  %1 = memref.alloc() : memref<128x64xf32>
  linalg.matmul {__byteir_scope_tile_anchor__} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%0 : memref<128x64xf32>)
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %0 : memref<128x64xf32>, memref<128x64xf32>) outs(%1 : memref<128x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg6: f32):  // no predecessors
    %3 = arith.addf %arg3, %arg4 : f32
    %4 = arith.mulf %3, %arg4 : f32
    linalg.yield %4 : f32
  }
  memref.dealloc %0 : memref<128x64xf32>
  %2 = memref.alloc() : memref<128x64xf32>
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : memref<128x64xf32>, memref<128x64xf32>) outs(%2 : memref<128x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg6: f32):  // no predecessors
    %3 = arith.addf %arg3, %arg4 : f32
    %4 = arith.mulf %3, %arg4 : f32
    linalg.yield %4 : f32
  }
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : memref<128x64xf32>, memref<128x64xf32>) outs(%arg2 : memref<128x64xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg6: f32):  // no predecessors
    %3 = arith.addf %arg3, %arg4 : f32
    %4 = arith.mulf %3, %arg4 : f32
    linalg.yield %4 : f32
  }
  memref.dealloc %1 : memref<128x64xf32>
  memref.dealloc %2 : memref<128x64xf32>
  return
}
// AXIS0-LABEL: func.func @matmul_element_static(
// AXIS0-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// AXIS0-DAG: %[[C128:.*]] = arith.constant 128 : index
// AXIS0-DAG: %[[C2:.*]] = arith.constant 2 : index
// AXIS0-DAG: %[[C0:.*]] = arith.constant 0 : index
// AXIS0: scf.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C128]]) step (%[[C2]]) {
// AXIS0:   memref.subview %[[ARG0]][%[[ARG3]], 0] [2, 64] [1, 1]
// AXIS0:   linalg.matmul
// AXIS0:   linalg.generic
// AXIS0:   linalg.generic
// AXIS0:   linalg.generic
// AXIS0:   scf.yield

// AXIS1-LABEL: func.func @matmul_element_static(
// AXIS1-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// AXIS1-DAG: %[[C64:.*]] = arith.constant 64 : index
// AXIS1-DAG: %[[C2:.*]] = arith.constant 2 : index
// AXIS1-DAG: %[[C0:.*]] = arith.constant 0 : index
// AXIS1: scf.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C2]]) {
// AXIS1:   memref.subview %[[ARG1]][0, %[[ARG3]]] [64, 2] [1, 1]
// AXIS1:   linalg.matmul
// AXIS1:   linalg.generic
// AXIS1:   linalg.generic
// AXIS1:   linalg.generic
// AXIS1:   scf.yield

// AXIS2-LABEL: func.func @matmul_element_static(
// AXIS2-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// AXIS2-DAG: %[[C64:.*]] = arith.constant 64 : index
// AXIS2-DAG: %[[C2:.*]] = arith.constant 2 : index
// AXIS2-DAG: %[[C0:.*]] = arith.constant 0 : index
// AXIS2: scf.for %[[ARG3:.*]] = %[[C0]] to %[[C64]] step %[[C2]] {
// AXIS2:   memref.subview %[[ARG0]][0, %[[ARG3]]] [128, 2] [1, 1]
// AXIS2:   memref.subview %[[ARG1]][%[[ARG3]], 0] [2, 64] [1, 1]
// AXIS2:   linalg.matmul
// AXIS2: }
// AXIS2: linalg.generic
// AXIS2: linalg.generic
// AXIS2: linalg.generic
