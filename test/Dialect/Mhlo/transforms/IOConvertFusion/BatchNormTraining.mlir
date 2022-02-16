// RUN: byteir-opt %s -fuse-io-convert="op-name=mhlo.batch_norm_training input-arg-idx=0 output-arg-idx=0 byre-compute-name=BatchNormTrainingOp" | FileCheck %s

func @batch_norm_training(%26: tensor<1x64x56x56xf16>, %arg6: tensor<64xf32>, %arg5: tensor<64xf32>) -> tensor<1x64x56x56xf16> {
    %27 = "mhlo.convert"(%26) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %28:3 = "mhlo.batch_norm_training"(%27, %arg6, %arg5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %29 = "mhlo.convert"(%28#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %29 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL: func @batch_norm_training
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}byre_compute_name = "BatchNormTrainingOp", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64
