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
