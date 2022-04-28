// RUN: byteir-opt %s -transform-layout="target-layout=NHWC" -split-input-file | FileCheck %s

func @batch_norm_training_fp16(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf16> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL: func @batch_norm_training_fp16
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.batch_norm_training{{.*}}feature_index = 3 : i64
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  return
