// RUN: byteir-opt %s -bounded-shape-infer | FileCheck %s

// CHECK-LABEL: @SameOperandsAndResultShape
func.func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32> {
  // CHECK-NEXT: byteir.bounded_shape0 = [8, 4]
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @InferShapedTypeOpInterface
func.func @InferShapedTypeOpInterface(%arg0 : tensor<8x4xi32>, %arg1 : tensor<8x4xi32>) -> tensor<?x4xi1> {
  // CHECK-NEXT: byteir.bounded_shape0 = [8, 4]
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<8x4xi32>, tensor<8x4xi32>) -> tensor<?x4xi1>
  return %0 : tensor<?x4xi1>
}

// CHECK-LABEL: @several_ops
func.func @several_ops(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  // CHECK-NEXT: byteir.bounded_shape0 = [8, 4]
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  // expected-warning@+1 {{inferReturnTypeComponents failed}}
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  // CHECK: mhlo.add {{.*}} {byteir.bounded_shape0 = [8, 4]}
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}

// CHECK-LABEL: @registered_shape_infer
func.func @registered_shape_infer(%arg0 : tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?xi64> {
  // CHECK-NEXT: byteir.bounded_shape0 = [32]
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.non_zero"} : (tensor<?x4xf32>) -> tensor<?xi64>
  return %0 : tensor<?xi64>
}

// CHECK-LABEL: @tf_where
func.func @tf_where(%arg0 : tensor<1xi1>) -> tensor<?x1xi64> {
  // CHECK-NEXT: byteir.bounded_shape0 = [1, 1]
  %0 = "mhlo.custom_call"(%arg0) { call_target_name = "tf.Where" } : (tensor<1xi1>) -> tensor<?x1xi64>
  return %0 : tensor<?x1xi64>
}