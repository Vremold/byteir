// RUN: byteir-opt %s -fuse-io-convert="op-name=mhlo.batch_norm_grad input-arg-idx=0,4 output-arg-idx=0 byre-compute-name=BatchNormGradOp" | FileCheck %s

func @batch_norm_grad(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %7: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %11, %9#1, %9#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
}
// CHECK-LABEL: func @batch_norm_grad
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_grad
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}byre_compute_name = "BatchNormGradOp", epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64
