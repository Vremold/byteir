// RUN: byteir-opt %s -linalg-scope-tile="axis=0 tile-size=2" -cse | FileCheck %s 

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_element_static_complete_0(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %0 = memref.alloc() : memref<128x64xf32>
  %1 = memref.alloc() : memref<128x64xf32>
  linalg.matmul {__byteir_scope_tile_axis__ = 0 : i32, __byteir_scope_tile_rank__ = 3 : i32} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%0 : memref<128x64xf32>)
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %0 : memref<128x64xf32>, memref<128x64xf32>) outs(%1 : memref<128x64xf32>) attrs = {__byteir_scope_tile_axis__ = 0 : i32, __byteir_scope_tile_rank__ = 2 : i32} {
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
// CHECK-LABEL: func.func @matmul_element_static_complete_0(
// CHECK-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C128]]) step (%[[C2]]) {
// CHECK:   memref.subview %[[ARG0]][%[[ARG3]], 0] [2, 64] [1, 1]
// CHECK:   linalg.matmul
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK: linalg.generic
// CHECK: linalg.generic

func.func @matmul_element_static_complete_1(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %0 = memref.alloc() : memref<128x64xf32>
  %1 = memref.alloc() : memref<128x64xf32>
  linalg.matmul ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%0 : memref<128x64xf32>)
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %0 : memref<128x64xf32>, memref<128x64xf32>) outs(%1 : memref<128x64xf32>) attrs = {__byteir_scope_tile_axis__ = 1 : i32, __byteir_scope_tile_rank__ = 2 : i32} {
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
// CHECK-LABEL: func.func @matmul_element_static_complete_1(
// CHECK-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// CHECK:   linalg.matmul
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.parallel (%[[ARG3:.*]]) = (%[[C0]]) to (%[[C64]]) step (%[[C2]]) {
// CHECK:   memref.subview %[[ARG0]][0, %[[ARG3]]] [128, 2] [1, 1]
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK: linalg.generic
// CHECK: linalg.generic

func.func @matmul_element_static_complete_2(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %0 = memref.alloc() : memref<128x64xf32>
  %1 = memref.alloc() : memref<128x64xf32>
  linalg.matmul {__byteir_scope_tile_axis__ = 2 : i32, __byteir_scope_tile_rank__ = 3 : i32} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%0 : memref<128x64xf32>)
  linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %0 : memref<128x64xf32>, memref<128x64xf32>) outs(%1 : memref<128x64xf32>){
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
// CHECK-LABEL: func.func @matmul_element_static_complete_2(
// CHECK-SAME: %[[ARG0:.*]]: memref<128x64xf32>, %[[ARG1:.*]]: memref<64x64xf32>, %[[ARG2:.*]]: memref<128x64xf32>
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[ARG3:.*]] = %[[C0]] to %[[C64]] step %[[C2]] {
// CHECK:   memref.subview %[[ARG0]][0, %[[ARG3]]] [128, 2] [1, 1]
// CHECK:   memref.subview %[[ARG1]][%[[ARG3]], 0] [2, 64] [1, 1]
// CHECK:   linalg.matmul
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
