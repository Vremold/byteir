// RUN: byteir-opt -convert-hlo-to-lhlo %s | FileCheck %s

func @batch_norm_training(%arg0 : tensor<1x576x768xf32>) -> tensor<1x576x768xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<576xf32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<576xf32>
  %2 = "mhlo.batch_norm_training"(%arg0, %1, %0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>) -> tuple<tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>>
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>>) -> tensor<1x576x768xf32>
  return %3 : tensor<1x576x768xf32>
}
// CHECK: lmhlo.batch_norm_training

func @clamp(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}
// CHECK: lmhlo.clamp

func @reduce_window_sum_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  return %0 : tensor<1x8x8x64xf32>
}
// CHECK: lmhlo.reduce_window

func @scatter(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):  // no predecessors
    %173 = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%173) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
  return %0: tensor<512x128xf32>
}
// CHECK: lmhlo.scatter

