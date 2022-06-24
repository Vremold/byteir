// RUN: byteir-opt %s -resolve-shape-constraint | FileCheck %s

func @meet_const(%arg0 : tensor<?x4xf32>, %arg1 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
  %a = shape.value_as_shape %0 : tensor<2xindex> -> !shape.shape
  %1 = shape.shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
  %b = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
  %c0 = arith.constant 0 : index
  %c = shape.const_size 1024
  %a0 = shape.get_extent %a, %c0 : !shape.shape, index -> !shape.size
  %b0 = shape.get_extent %b, %c0 : !shape.shape, index -> !shape.size
  %sum = shape.add %a0, %b0 : !shape.size, !shape.size -> !shape.size
  "shape_ext.meet"(%c, %sum) : (!shape.size, !shape.size) -> ()
  %result = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  "shape_ext.tie"(%result, %sum) : (tensor<?x4xf32>, !shape.size) -> ()
  return %result : tensor<?x4xf32>
}
// CHECK-LABEL: @meet_const(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<1024x4xf32>
// CHECK-NEXT: %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<1024x4xf32>
// CHECK-NEXT: return %0 : tensor<1024x4xf32>