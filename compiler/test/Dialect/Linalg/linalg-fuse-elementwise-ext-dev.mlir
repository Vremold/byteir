// RUN: byteir-opt %s -linalg-fuse-elementwise-ext -split-input-file | FileCheck %s

// this is called dev, since it is not perfect yet.

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#trait = {
  indexing_maps = [#map, #map],
  iterator_types = ["parallel", "parallel"]
}
func.func @break_outs_dependency(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic #trait ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %1 = arith.addf %arg1, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?xf32>
  %2 = linalg.generic #trait ins(%0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %3 = arith.mulf %arg1, %arg1 : f32
         linalg.yield %3 : f32
       } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func @break_outs_dependency(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]])
//      CHECK:   %[[GENERIC1:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[GENERIC1]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[GENERIC1]], %[[C1]]
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]])
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)
