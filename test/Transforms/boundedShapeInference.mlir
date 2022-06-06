// RUN: byteir-opt %s -bounded-shape-infer | FileCheck %s

// CHECK-LABEL: @SameOperandsAndResultShape
func @SameOperandsAndResultShape(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?x4xf32> {
  // CHECK-NEXT: byteir.bounded_shape0 = [8, 4]
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @InferShapedTypeOpInterface
func @InferShapedTypeOpInterface(%pred : tensor<i1>, %a : tensor<?x2xf32> {byteir.bounded_shape = [4, 2]}, %b : tensor<?x2xf32> {byteir.bounded_shape = [4, 2]}) -> tensor<?x2xf32> {
  // CHECK-NEXT: byteir.bounded_shape0 = [4, 2]
  %0 = "mhlo.select"(%pred, %a, %b) : (tensor<i1>, tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: @several_ops
func @several_ops(%arg0: tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
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
func @registered_shape_infer(%arg0 : tensor<?x4xf32> {byteir.bounded_shape = [8, 4]}) -> tensor<?xi64> {
  // CHECK-NEXT: byteir.bounded_shape0 = [32]
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "NonZero"} : (tensor<?x4xf32>) -> tensor<?xi64>
  return %0 : tensor<?xi64>
}
