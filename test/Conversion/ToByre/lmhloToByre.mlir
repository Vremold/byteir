// RUN: byteir-opt -convert-to-byre --canonicalize %s | FileCheck %s

module {
// CHECK: module attributes {byre.container_module}  {
  func @mhlo_add(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK:   func @mhlo_add(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @AddOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) : memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

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

  func @mhlo_add_no_annotation(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> attributes { __placeholder__byre.entry_point} {
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK:   func @mhlo_add_no_annotation(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "Input1", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @AddOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]]) : memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

  func @mhlo_add_splat_const(%arg0: memref<4xf32>) -> memref<4xf32> attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.constant"(%0) {value = dense<2.000000e+00> : tensor<4xf32>, name = "two"} : (memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK-LABEL: func @mhlo_add_splat_const
//   CHECK-SAME: %[[ARG_0:.*]]: memref<4xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}
//   CHECK_SAME: %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}
// CHECK-NEXT: %[[MEM_0:.*]] = memref.alloc()
// CHECK-NEXT: byre.compute @FillOp(%[[MEM_0]])
// CHECK-NEXT: byre.compute @AddOp(%[[ARG_0]], %[[MEM_0]], %[[ARG_1]])
// CHECK-NEXT: return

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

  func @mhlo_scatter(%arg0: memref<512x128xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<128x1xi64> {__placeholder__byre.argname = "B"}, %arg2: memref<128x128xf32> {__placeholder__byre.argname = "C"}) -> (memref<512x128xf32> {__placeholder__byre.argname = "D"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%arg0, %arg1, %arg2, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %1 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    return %0 : memref<512x128xf32>
  }
  // CHECK-LABEL: mhlo_scatter
  // CHECK-NEXT: byre.compute @IndexPutOp

  func @mhlo_gather(%arg0: memref<30522x128xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<128xui32> {__placeholder__byre.argname = "B"}) -> (memref<128x128xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %2 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg0, %arg1, %2) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    return %2 : memref<128x128xf32>
  }
  // CHECK-LABEL: mhlo_gather
  // CHECK-NEXT: byre.compute @IndexSelectOp

  func @mhlo_slice(%arg0: memref<1x512xi64> {__placeholder__byre.argname = "A"}) -> (memref<1x128xi64> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg0, %0) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    return %0 : memref<1x128xi64>
  }
  // CHECK-LABEL: mhlo_slice
  // CHECK-NEXT: byre.compute @AliasOp

  func @mhlo_reshape(%arg0: memref<1x1024xi64>) -> (memref<32x32xi64>) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<32x32xi64>
    "lmhlo.reshape"(%arg0, %0) : (memref<1x1024xi64>, memref<32x32xi64>) -> ()
    return %0 : memref<32x32xi64>
  }
  // CHECK-LABEL: mhlo_reshape
  // CHECK-NEXT: byre.compute @AliasOp

  func @mhlo_reduce(%arg0: memref<1x128x128xf32> {__placeholder__byre.argname = "A"}) -> (memref<128xf32> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<128xf32>
    "lmhlo.reduce"(%arg0, %0, %1) ( {
    ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<1x128x128xf32>, memref<f32>, memref<128xf32>) -> ()
    return %1 : memref<128xf32>
  }
  // CHECK-LABEL: mhlo_reduce
  // CHECK-NEXT: byre.compute @ReduceSumOp(%arg0, %arg1)
  //   CHECK-DAG: dimensions = dense<[0, 1]>
}
