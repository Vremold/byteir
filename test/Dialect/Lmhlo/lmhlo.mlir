// RUN: byteir-opt %s | FileCheck %s

func @lmhlo_add(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  return 
}
// CHECK-LABEL: func @lmhlo_add

func @mhlo_add_constant(%arg0: memref<4xf32>) -> memref<4xf32> {
  %0 = memref.alloc() : memref<4xf32>
  "lmhlo.constant"(%0) {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>, name = "weight1"} : (memref<4xf32>) -> ()
  %1 = memref.alloc() : memref<4xf32>
  "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
  return %1 : memref<4xf32>
}
// CHECK-LABEL: func @mhlo_add_constant
