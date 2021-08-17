// RUN: byteir-opt %s | FileCheck %s

func @reduce_window(%arg: memref<112x112xf32>,
             %init: memref<f32>,
             %result: memref<56x56xf32>) {
  "lmhlo.reduce_window"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.maximum"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {
      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
      window_dimensions = dense<[3, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 2]> : tensor<2xi64>
    } : (memref<112x112xf32>, memref<f32>, memref<56x56xf32>) -> ()
  return
}
// CHECK-LABEL: func @reduce_window

