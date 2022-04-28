// RUN: byteir-opt %s -transform-layout="target-layout=NHWC" -split-input-file | FileCheck %s

func @conv(%arg0: tensor<5x69x31x95xf32>, %arg1: tensor<64x69x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func @conv
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.convolution{{.*}}dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NEXT:  mhlo.transpose