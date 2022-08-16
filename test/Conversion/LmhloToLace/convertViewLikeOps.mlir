// RUN: byteir-opt %s --lmhlo-to-lace | FileCheck %s

func.func @convert_reshape_static(%arg0: memref<2x3xf32>) -> memref<6xf32>  {
  %0 = memref.alloc() : memref<6xf32>
  "lmhlo.reshape"(%arg0, %0) : (memref<2x3xf32>, memref<6xf32>) -> ()
  return %0: memref<6xf32>
}
// CHECK-LABEL: func.func @convert_reshape_static
//   CHECK: lace.reshape

func.func @convert_slice_static(%arg0: memref<2x3xf32>) -> memref<1x3xf32> {
  %0 = memref.alloc() : memref<1x3xf32>
  "lmhlo.slice"(%arg0, %0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<2x3xf32>, memref<1x3xf32>) -> ()
  return %0: memref<1x3xf32>
}
// CHECK-LABEL: func.func @convert_slice_static
//   CHECK: lace.slice

func.func @convert_slice_to_arg(%arg0: memref<2x3xf32>, %arg1: memref<1x3xf32>) -> () {
  "lmhlo.slice"(%arg0, %arg1) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<2x3xf32>, memref<1x3xf32>) -> ()
  return
}
// CHECK-LABEL: func.func @convert_slice_to_arg
//   CHECK-NEXT: lace.slice
//   CHECK-NEXT: memref.copy
