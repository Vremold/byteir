// RUN: byteir-opt %s -static-shape-infer | FileCheck %s

// CHECK-LABEL: @InferShapedTypeOpInterface
func @InferShapedTypeOpInterface(%pred : tensor<i1>, %a : tensor<4x2xf32>, %b : tensor<4x2xf32>) -> tensor<?x2xf32> {
  // CHECK-NEXT: %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %0 = "mhlo.select"(%pred, %a, %b) : (tensor<i1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<?x2xf32>
  // CHECK-NEXT: return %0 : tensor<4x2xf32>
  return %0 : tensor<?x2xf32>
}