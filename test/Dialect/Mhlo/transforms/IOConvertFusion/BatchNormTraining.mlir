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
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"

func @batch_norm_training_grad(%118: tensor<32x256x14x14xf16>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %303: tensor<32x256x14x14xf16>, %306: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) {
  %119 = "mhlo.convert"(%118) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %120:3 = "mhlo.batch_norm_training"(%119, %arg55, %arg54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %121 = "mhlo.convert"(%120#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
  %307 = "mhlo.convert"(%303) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %308:3 = "mhlo.batch_norm_grad"(%119, %arg55, %120#1, %306, %307) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %309 = "mhlo.convert"(%308#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
  return %121, %309 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>
}
// CHECK-LABEL: func @batch_norm_training_grad
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.batch_norm_grad
// CHECK-NEXT:  mhlo.convert