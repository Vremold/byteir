// RUN: byteir-opt %s -hlo-fold | FileCheck %s

func @broadcast_transpose(%arg0 : tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func @broadcast_transpose
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<1> : tensor<1xi64>}{{.*}}
// CHECK:  return

func @broadcast_transpose_non_dim(%arg0 : tensor<f32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func @broadcast_transpose_non_dim
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<> : tensor<0xi64>}{{.*}}
// CHECK:  return

func @broadcast_transpose_multi_dim(%arg0 : tensor<95x64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<95x64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func @broadcast_transpose_multi_dim
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<[3, 1]> : tensor<2xi64>}{{.*}}
// CHECK:  return

func @transpose_transpose(%arg0 : tensor<31x20x32xf32>) -> tensor<20x32x31xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<20x32x31xf32>
    return %1 : tensor<20x32x31xf32>
}
// CHECK-LABEL: func @transpose_transpose
// CHECK-NEXT:  mhlo.transpose{{.*}}{permutation = dense<[1, 2, 0]> : tensor<3xi64>}
// CHECK:  return

func @transpose_transpose_to_noop(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %1 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func @transpose_transpose_to_noop
// CHECK:  return %arg0 : tensor<31x20x32xf32>

func @add_scatteradd_right(%arg0 : tensor<30522x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %2 = mhlo.add %arg0, %1 : tensor<30522x128xf32>
    return %2 : tensor<30522x128xf32>
}
// CHECK-LABEL: func @add_scatteradd_right
// CHECK-NEXT: mhlo.scatter

func @add_scatteradd_left(%arg0 : tensor<30522x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %2 = mhlo.add %1, %arg0 : tensor<30522x128xf32>
    return %2 : tensor<30522x128xf32>
}
// CHECK-LABEL: func @add_scatteradd_left
// CHECK-NEXT: mhlo.scatter
