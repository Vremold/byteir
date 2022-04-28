// RUN: byteir-opt %s -transform-layout="target-layout=NDHWC" -split-input-file | FileCheck %s

func @conv3d(%140: tensor<1x3x100x27x48xf32>, %16: tensor<32x3x1x3x3xf32>) -> tensor<1x32x100x27x48xf32> {
  %141 = "mhlo.convolution"(%140, %16) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2]>, feature_group_count = 1 : i64, padding = dense<[[0, 0], [1, 1], [1, 1]]> : tensor<3x2xi64>, rhs_dilation = dense<1> : tensor<3xi64>, window_strides = dense<1> : tensor<3xi64>} : (tensor<1x3x100x27x48xf32>, tensor<32x3x1x3x3xf32>) -> tensor<1x32x100x27x48xf32>
  return %141 : tensor<1x32x100x27x48xf32>
}
// CHECK-LABEL: func @conv3d
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.convolution{{.*}}dim_numbers = [b, 0, 1, 2, f]x[o, 0, 1, 2, i]->[b, 0, 1, 2, f]
// CHECK-NEXT:  mhlo.transpose