// RUN: byteir-opt %s --lmhlo-to-memref | FileCheck %s

func.func @convert_reshape_static() {
  %c8 = arith.constant 8 : index
  %0 = memref.alloc() : memref<1024xf32>
  %1 = memref.alloc() : memref<16x64xf32>
// CHECK: %1 = memref.alloc() : memref<2xi64>
  "lmhlo.reshape"(%0, %1) : (memref<1024xf32>, memref<16x64xf32>) -> ()
// CHECK: "lmhlo.constant"(%1) {value = dense<[16, 64]> : tensor<2xi64>} : (memref<2xi64>) -> ()
// CHECK: %2 = memref.reshape %0(%1) : (memref<1024xf32>, memref<2xi64>) -> memref<16x64xf32>
  %2 = memref.load %0[%c8] : memref<1024xf32>
  %3 = arith.addf %2, %2 : f32
  memref.store %3, %1[%c8, %c8] : memref<16x64xf32>
// CHECK: memref.store %4, %2
  return
}

func.func @convert_slice_static() {
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() : memref<128x100x32xf32>
  %1 = memref.alloc() : memref<64x30x8xf32>
  "lmhlo.slice"(%0, %1) {limit_indices = dense<[64, 60, 32]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<[1, 2, 4]> : tensor<3xi64>} : (memref<128x100x32xf32>, memref<64x30x8xf32>) -> ()
// CHECK: %1 = memref.subview %0
  %2 = memref.load %0[%c4, %c4, %c4] : memref<128x100x32xf32>
  %3 = arith.addf %2, %2 : f32
  memref.store %3, %1[%c4, %c4, %c4] : memref<64x30x8xf32>
  return
}


