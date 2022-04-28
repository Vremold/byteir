// RUN: byteir-opt %s -transform-layout="target-layout=NHWC" -split-input-file | FileCheck %s

func @max_pool_NCHW(%181: tensor<1x32x128x128xf32>) -> tensor<1x32x64x64xf32> {
  %163 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %182 = "mhlo.reduce_window"(%181, %163) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %522 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%522) : (tensor<f32>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [0, 1], [0, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x32x128x128xf32>, tensor<f32>) -> tensor<1x32x64x64xf32>
  return %182 : tensor<1x32x64x64xf32>
}
// CHECK-LABEL: func @max_pool_NCHW
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.reduce_window
// CHECK:  mhlo.transpose

// -----

func @avg_pool_NCHW(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x2x3xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<4.000000e+00> : tensor<1x1x2x3xf32>
  %2 = "mhlo.reduce_window"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<1x1x3x4xf32>, tensor<f32>) -> tensor<1x1x2x3xf32>
  %3 = mhlo.divide %2, %1 : tensor<1x1x2x3xf32>
  return %3 : tensor<1x1x2x3xf32>
}
// CHECK-LABEL: func @avg_pool_NCHW
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.reduce_window
// CHECK:  mhlo.transpose
// CHECK-NEXT:  mhlo.divide