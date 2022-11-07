// RUN: byteir-opt %s | FileCheck %s

module attributes {byre.container_module}  {
  func.func @main(%arg0: memref<2x3xf32> {byre.argname = "input_B", byre.argtype = 1 : i32}, %arg1: memref<2x3xf32> {byre.argname = "input_A", byre.argtype = 1 : i32}, %arg2: memref<2x2xf32> {byre.argname = "grad_C", byre.argtype = 1 : i32}, %arg3: memref<2x2xf32> {byre.argname = "output_C", byre.argtype = 2 : i32}, %arg4: memref<2x3xf32> {byre.argname = "grad_A", byre.argtype = 2 : i32}, %arg5: memref<2x3xf32> {byre.argname = "grad_B", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @ftv4.matmul(%arg1, %arg0, %arg3) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true} : memref<2x3xf32>, memref<2x3xf32>, memref<2x2xf32>
    byre.compute @ftv4.matmul_backward(%arg2, %arg1, %arg0, %arg4, %arg5) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = true} : memref<2x2xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>
    return
  }
}

// CHECK-LABEL: func.func @main
