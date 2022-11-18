// RUN: byteir-opt %s -convert-to-byre -cse | FileCheck %s

module  {
  func.func @main(%arg0: memref<2x3xf32> {__placeholder__byre.argname = "input_B", __placeholder__byre.argtype = 1 : i32}, %arg1: memref<2x3xf32> {__placeholder__byre.argname = "input_A", __placeholder__byre.argtype = 1 : i32}, %arg2: memref<2x2xf32> {__placeholder__byre.argname = "grad_C", __placeholder__byre.argtype = 1 : i32}) -> (memref<2x2xf32>, memref<2x3xf32>, memref<2x3xf32>) attributes {__placeholder__byre.entry_point, __placeholder__byre.result_attrs = [{__placeholder__byre.argname = "output_C", __placeholder__byre.argtype = 2 : i32}, {__placeholder__byre.argname = "grad_A", __placeholder__byre.argtype = 2 : i32}, {__placeholder__byre.argname = "grad_B", __placeholder__byre.argtype = 2 : i32}]} {
    %0 = memref.alloc() : memref<2x2xf32>
    "lmhlo.custom_call"(%arg1, %arg0, %0) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = array<i32: 2, 1>} : (memref<2x3xf32>, memref<2x3xf32>, memref<2x2xf32>) -> ()
    %1 = memref.alloc() : memref<2x3xf32>
    %2 = memref.alloc() : memref<2x3xf32>
    "lmhlo.custom_call"(%arg2, %arg1, %arg0, %1, %2) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>) -> ()
    return %0, %1, %2 : memref<2x2xf32>, memref<2x3xf32>, memref<2x3xf32>
  }
}

// CHECK-LABEL: func.func @main
//  CHECK-NEXT: byre.compute
//  CHECK-NEXT: byre.compute
//  CHECK-NEXT: return
