// RUN: byteir-opt -convert-hlo-to-lhlo %s | FileCheck %s

func @scatter(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):  // no predecessors
    %173 = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%173) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
  return %0: tensor<512x128xf32>
}
// CHECK: lmhlo.scatter
