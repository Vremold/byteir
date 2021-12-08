// RUN: byteir-opt --convert-hlo-to-lhlo %s | FileCheck %s
  
func @aten__index_select.148(%arg0: tensor<30522x128xf32>, %arg1: tensor<128xui32>) -> tensor<128x128xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK: lmhlo.gather