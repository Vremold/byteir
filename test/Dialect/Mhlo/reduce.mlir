// RUN: byteir-opt %s | FileCheck %s

func @reduce(%arg0: tensor<1x8xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
// CHECK-LABEL: func @reduce

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
// CHECK-LABEL: func @reduce_window_sum_nhwc

// func @reduce_window_multiple_operand(%arg0: tensor<1x17x17x64xf32>, %arg1: tensor<1x17x17x64xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> 
//   (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
//   %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %arg2) ( {
//   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
//     %1 = mhlo.add %arg4, %arg6 : tensor<f32>
//     %2 = mhlo.add %arg5, %arg6 : tensor<f32>
//     %3 = "mhlo.tuple"(%1, %2) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
//     "mhlo.return"(%3) : (tuple<tensor<f32>, tensor<f32>>) -> ()
//   }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}
//     : (tensor<1x17x17x64xf32>, tensor<1x17x17x64xf32>, tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
//   return %0#0, %0#1 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
// }