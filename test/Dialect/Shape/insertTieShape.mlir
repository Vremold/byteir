// RUN: byteir-opt %s -insert-tie-shape | FileCheck %s

func.func @simple(%arg0: tensor<?x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}
// CHECK-LABEL: func.func @simple(%arg0: tensor<?x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
// CHECK-NEXT:     %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = tensor.dim %0, %c0 : tensor<?x4xf32>
// CHECK-NEXT:     "shape_ext.tie"(%0, %1) : (tensor<?x4xf32>, index) -> ()
// CHECK-NEXT:     %2 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-NEXT:     %3 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT:     %c0_0 = arith.constant 0 : index
// CHECK-NEXT:     %4 = tensor.dim %3, %c0_0 : tensor<?x4xf32>
// CHECK-NEXT:     "shape_ext.tie"(%3, %4) : (tensor<?x4xf32>, index) -> ()
// CHECK-NEXT:     %5 = mhlo.add %0, %3 : tensor<?x4xf32>
// CHECK-NEXT:     %c0_1 = arith.constant 0 : index
// CHECK-NEXT:     %6 = tensor.dim %5, %c0_1 : tensor<?x4xf32>
// CHECK-NEXT:     "shape_ext.tie"(%5, %6) : (tensor<?x4xf32>, index) -> ()
// CHECK-NEXT:     return %5 : tensor<?x4xf32>
// CHECK-NEXT:   }
