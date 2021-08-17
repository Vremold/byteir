// RUN: byteir-opt --convert-hlo-to-lhlo %s | FileCheck %s

func @batch_norm_training(%arg0 : tensor<1x576x768xf32>) -> tensor<1x576x768xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<576xf32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<576xf32>
  %2 = "mhlo.batch_norm_training"(%arg0, %1, %0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>) -> tuple<tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>>
  %3 = "mhlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>>) -> tensor<1x576x768xf32>
  return %3 : tensor<1x576x768xf32>
}
// CHECK: lmhlo.batch_norm_training
