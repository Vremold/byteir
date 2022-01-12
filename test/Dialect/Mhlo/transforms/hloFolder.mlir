// RUN: byteir-opt %s -hlo-fold | FileCheck %s

func @add_scatteradd_right(%arg0 : tensor<30522x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %2 = mhlo.add %arg0, %1 : tensor<30522x128xf32>
    return %2 : tensor<30522x128xf32>
}
// CHECK-LABEL: func @add_scatteradd_right
// CHECK-NEXT: mhlo.scatter

func @add_scatteradd_left(%arg0 : tensor<30522x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %2 = mhlo.add %1, %arg0 : tensor<30522x128xf32>
    return %2 : tensor<30522x128xf32>
}
// CHECK-LABEL: func @add_scatteradd_left
// CHECK-NEXT: mhlo.scatter
