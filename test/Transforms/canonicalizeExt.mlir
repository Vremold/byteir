// RUN: byteir-opt %s -canonicalize-ext | FileCheck %s

func.func @dead_custom_call() -> tensor<128xf32> {
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %1 = "mhlo.custom_call"(%c0) {backend_config = "", call_target_name = "foo", has_side_effect = false} : (tensor<128xf32>) -> tensor<128xf32>
  return %c0: tensor<128xf32>
}
// CHECK-LABEL: dead_custom_call
// CHECK-NOT: mhlo.custom_call

func.func @eliminate_splat_constant_transpose() -> tensor<2x1x4x3xi32> {
  %0 = mhlo.constant dense<0> : tensor<1x2x3x4xi32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %1: tensor<2x1x4x3xi32>
}
// CHECK-LABEL: eliminate_splat_constant_transpose
// CHECK-NEXT: %0 = mhlo.constant dense<0> : tensor<2x1x4x3xi32>

func.func @fold_useless_shape_broadcast(%arg0: tensor<?x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %0 = shape.const_shape [4] : tensor<1xindex>
  %1 = mhlo.constant dense<[[-0.570340514, 0.117151208, -0.135694504, -1.57919896], [0.520053327, 0.762166619, 0.322875232, -1.69871449], [-1.26622009, 0.63558042, 5.698780e-01, 0.954656243], [0.776482939, 0.348752886, 2.03235912, 0.837243676]]> : tensor<4x4xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %3 = shape.shape_of %2 : tensor<?x4xf32> -> tensor<2xindex>
  %4 = shape.broadcast %3, %0 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%2, %4) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  return %5 : tensor<?x4xf32>
}
// CHECK-LABEL: fold_useless_shape_broadcast
// CHECK-NOT: shape.broadcast

// FIXME: make constant really large or trigger canonicalize-ext anywhy.
func.func @fold_large_constant_binary_op() -> tensor<2xf32> {
  %0 = mhlo.constant dense<[0.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[1.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %2 = mhlo.add %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}
// CHECK-NOT: mhlo.add
// CHECK: mhlo.constant dense<[1.000000e+00, 2.000000e+00]>

func.func @fold_concat_of_continuous_slices(%arg0: tensor<4x11xf32>) -> tensor<4x11xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 7]> : tensor<2xi64>, start_indices = dense<[0, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x2xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 11]> : tensor<2xi64>, start_indices = dense<[0, 7]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x5xf32>
  %3 = "mhlo.concatenate"(%2, %0, %1) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>, tensor<4x4xf32>) -> tensor<4x11xf32>
  return %3 : tensor<4x11xf32> 
}
// CHECK-LABEL: func.func @fold_concat_of_continuous_slices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x11xf32>)
// CHECK-NEXT: return %[[ARG0]] : tensor<4x11xf32>
