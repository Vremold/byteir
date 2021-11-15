// RUN: byteir-opt --convert-to-byre %s | FileCheck %s

module {
// CHECK: module attributes {byre.container_module}  {
  func @mhlo_add_weight(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}) -> (memref<4xf32> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.constant"(%0) {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>, name = "weight1"} : (memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK:   func @mhlo_add_weight(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "weight1", byre.argtype = 4 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @AddOp(%[[ARG_1]], %[[ARG_0]], %[[ARG_2]]) : memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

  func @mhlo_add(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK:   func @mhlo_add(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @AddOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) : memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

  func @mhlo_matmul(%arg0: memref<128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>) -> ()
    return %0 : memref<128x32xf32>
  }
// CHECK:   func @mhlo_matmul(%[[ARG_0:.*]]: memref<128x64xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<64x32xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<128x32xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @MatmulOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>
// CHECK:     return

  func @mhlo_matmul1(%arg0: memref<128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>) -> ()
    return %0 : memref<128x32xf32>
  }
// CHECK:   func @mhlo_matmul1(%[[ARG_0:.*]]: memref<128x64xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<64x32xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<128x32xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @MatmulOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>
// CHECK:     return


  func @mhlo_batch_matmul(%arg0: memref<3x128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<3x64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<3x128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<3x128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<3x128x64xf32>, memref<3x64x32xf32>, memref<3x128x32xf32>) -> ()
    return %0 : memref<3x128x32xf32>
  }
// CHECK:   func @mhlo_batch_matmul(%[[ARG_0:.*]]: memref<3x128x64xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<3x64x32xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<3x128x32xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @BatchMatmulOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) {lhs_batching_dimensions = [0], lhs_contracting_dimension = 1 : i64, rhs_batching_dimensions = [0], rhs_contracting_dimension = 0 : i64} : memref<3x128x64xf32>, memref<3x64x32xf32>, memref<3x128x32xf32>
// CHECK:     return

}
