// RUN: byteir-opt -expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse %s | FileCheck %s

module  {
  func.func @main(%arg0: tensor<2x3xf32> {__placeholder__byre.argname = "input_B", __placeholder__byre.argtype = 1 : i32}, %arg1: tensor<2x3xf32> {__placeholder__byre.argname = "input_A", __placeholder__byre.argtype = 1 : i32}, %arg2: tensor<2x2xf32> {__placeholder__byre.argname = "grad_C", __placeholder__byre.argtype = 1 : i32}) -> tuple<tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>> attributes {__placeholder__byre.entry_point, __placeholder__byre.result_attrs = [{__placeholder__byre.argname = "output_C", __placeholder__byre.argtype = 2 : i32}, {__placeholder__byre.argname = "grad_A", __placeholder__byre.argtype = 2 : i32}, {__placeholder__byre.argname = "grad_B", __placeholder__byre.argtype = 2 : i32}]} {
    %0 = "mhlo.custom_call"(%arg1, %arg0) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
    %1 = "mhlo.custom_call"(%arg2, %arg1, %arg0) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tuple<tensor<2x3xf32>, tensor<2x3xf32>>
    %2 = "mhlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
    %3 = "mhlo.get_tuple_element"(%1) {index = 1 : i32} : (tuple<tensor<2x3xf32>, tensor<2x3xf32>>) -> tensor<2x3xf32>
    %4 = "mhlo.tuple"(%0, %2, %3) : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tuple<tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>>
    return %4 : tuple<tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>>
  }
}

//  CHECK: return %{{.*}} : tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x3xf32>