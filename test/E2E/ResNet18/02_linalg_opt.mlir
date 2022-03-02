// RUN: byteir-opt %s -linalg-opt | FileCheck %s

// CHECK-LABEL: func @main
module {
  func private @Unknown0(%arg0: tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
    return %0 : tensor<32x3x224x224xf16>
  }
  func private @Unknown1(%arg0: tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    return %0 : tensor<64x3x7x7xf16>
  }
  func private @BatchNormTrainingOp2(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    return %2, %1#1, %1#2 : tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @Unknown3(%arg0: tensor<1000x512xf32>) -> tensor<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func private @Unknown4(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func private @Unknown5(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func private @Unknown6(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func private @Unknown7(%arg0: tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    return %0 : tensor<64x64x3x3xf16>
  }
  func private @Unknown8(%arg0: tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    return %0 : tensor<128x64x3x3xf16>
  }
  func private @Unknown9(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func private @Unknown10(%arg0: tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    return %0 : tensor<128x64x1x1xf16>
  }
  func private @Unknown11(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func private @Unknown12(%arg0: tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    return %0 : tensor<128x128x3x3xf16>
  }
  func private @Unknown13(%arg0: tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    return %0 : tensor<256x128x3x3xf16>
  }
  func private @Unknown14(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func private @Unknown15(%arg0: tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    return %0 : tensor<256x128x1x1xf16>
  }
  func private @Unknown16(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func private @Unknown17(%arg0: tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    return %0 : tensor<256x256x3x3xf16>
  }
  func private @Unknown18(%arg0: tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    return %0 : tensor<512x256x3x3xf16>
  }
  func private @Unknown19(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func private @Unknown20(%arg0: tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    return %0 : tensor<512x256x1x1xf16>
  }
  func private @Unknown21(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>
  }
  func private @Unknown23(%arg0: tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<32x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x112x112xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x64x112x112xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xi1>
    return %1, %2 : tensor<32x64x112x112xf16>, tensor<32x64x112x112xi1>
  }
  func private @BatchNormTrainingOp24(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %2, %1#1, %1#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @Unknown25(%arg0: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x64x56x56xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    return %1, %2 : tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp26(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %2, %1#1, %1#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @Unknown27(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x64x56x56xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    return %2, %3 : tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp28(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %2, %1#1, %1#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @Unknown29(%arg0: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x64x56x56xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    return %1, %2 : tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp30(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %2, %1#1, %1#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @Unknown31(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x64x56x56xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    return %2, %3 : tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp32(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %2, %1#1, %1#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @BatchNormTrainingOp33(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %2, %1#1, %1#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @Unknown34(%arg0: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x128x28x28xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    return %1, %2 : tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp35(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %2, %1#1, %1#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @Unknown36(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x128x28x28xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    return %2, %3 : tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp37(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %2, %1#1, %1#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @Unknown38(%arg0: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x128x28x28xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    return %1, %2 : tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp39(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %2, %1#1, %1#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @Unknown40(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x128x28x28xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    return %2, %3 : tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp41(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %2, %1#1, %1#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @BatchNormTrainingOp42(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %2, %1#1, %1#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @Unknown43(%arg0: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x256x14x14xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    return %1, %2 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp44(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %2, %1#1, %1#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @Unknown45(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x256x14x14xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    return %2, %3 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp46(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %2, %1#1, %1#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @Unknown47(%arg0: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x256x14x14xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    return %1, %2 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp48(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %2, %1#1, %1#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @Unknown49(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x256x14x14xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    return %2, %3 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp50(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %2, %1#1, %1#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @BatchNormTrainingOp51(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %2, %1#1, %1#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown52(%arg0: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x512x7x7xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    return %1, %2 : tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp53(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %2, %1#1, %1#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown54(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x512x7x7xf16>
    %2 = mhlo.maximum %1, %0 : tensor<32x512x7x7xf16>
    %3 = "mhlo.compare"(%2, %0) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    return %2, %3 : tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp55(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %2, %1#1, %1#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown56(%arg0: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = mhlo.maximum %arg0, %0 : tensor<32x512x7x7xf16>
    %2 = "mhlo.compare"(%1, %0) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    return %1, %2 : tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp57(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %2, %1#1, %1#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown58(%arg0: tensor<32x512xf16>, %arg1: tensor<32x512x7x7xf16>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.900000e+01> : tensor<32x512x7x7xf16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<32x512xf16>) -> tensor<32x512x7x7xf16>
    %3 = mhlo.divide %2, %0 : tensor<32x512x7x7xf16>
    %4 = mhlo.add %arg1, %arg2 : tensor<32x512x7x7xf16>
    %5 = mhlo.maximum %4, %1 : tensor<32x512x7x7xf16>
    %6 = "mhlo.compare"(%5, %1) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %7 = "mhlo.select"(%6, %3, %1) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %5, %7 : tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>
  }
  func private @Unknown59(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp60(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp61(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp62(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown63(%arg0: tensor<32x512x7x7xi1>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %1 : tensor<32x512x7x7xf16>
  }
  func private @Unknown64(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp65(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp66(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp67(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown68(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>, %arg2: tensor<32x512x7x7xi1>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x512x7x7xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @Unknown69(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp70(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp71(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp72(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown73(%arg0: tensor<32x512x7x7xi1>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %1 : tensor<32x512x7x7xf16>
  }
  func private @Unknown74(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp75(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp76(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp77(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func private @Unknown78(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp79(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func private @Unknown82(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @Unknown83(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp84(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp85(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp86(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown87(%arg0: tensor<32x256x14x14xi1>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @Unknown88(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp89(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp90(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp91(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown92(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @Unknown93(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp94(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown97(%arg0: tensor<32x256x14x14xi1>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @Unknown98(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp99(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp100(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp101(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func private @Unknown102(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp103(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp104(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp105(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func private @Unknown106(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @Unknown107(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp108(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp109(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp110(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown111(%arg0: tensor<32x128x28x28xi1>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @Unknown112(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp113(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp114(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp115(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown116(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @Unknown117(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp118(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp119(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp120(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown121(%arg0: tensor<32x128x28x28xi1>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @Unknown122(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp123(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp124(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp125(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func private @Unknown126(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp127(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp128(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp129(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func private @Unknown130(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @Unknown131(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp132(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp133(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp134(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown135(%arg0: tensor<32x64x56x56xi1>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @Unknown136(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp137(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp138(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp139(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown140(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @Unknown141(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp142(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp143(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp144(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown145(%arg0: tensor<32x64x56x56xi1>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @Unknown146(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp147(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp148(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp149(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown150(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    return %0 : tensor<32x64x56x56xf16>
  }
  func private @Unknown151(%arg0: tensor<32x64x112x112xi1>, %arg1: tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x112x112xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    return %1 : tensor<32x64x112x112xf16>
  }
  func private @Unknown152(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp153(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    return %3, %2#1, %2#2 : tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardFilterOp154(%arg0: tensor<32x3x224x224xf16>, %arg1: tensor<32x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func private @Unknown155(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    return %0 : tensor<64x3x7x7xf32>
  }
  func private @Unknown156(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown157(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown158(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown159(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown160(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    return %0 : tensor<128x64x3x3xf32>
  }
  func private @Unknown161(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown162(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    return %0 : tensor<128x64x1x1xf32>
  }
  func private @Unknown163(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown164(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown165(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    return %0 : tensor<256x128x3x3xf32>
  }
  func private @Unknown166(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown167(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    return %0 : tensor<256x128x1x1xf32>
  }
  func private @Unknown168(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown169(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown170(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    return %0 : tensor<512x256x3x3xf32>
  }
  func private @Unknown171(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown172(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    return %0 : tensor<512x256x1x1xf32>
  }
  func private @Unknown173(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown174(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown175(%arg0: tensor<32x512xf16>) -> tensor<32x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<2.040100e-02> : tensor<32x512xf16>
    %1 = mhlo.multiply %arg0, %0 : tensor<32x512xf16>
    return %1 : tensor<32x512xf16>
  }
  func private @MatmulOp176(%arg0: tensor<32x512xf16>, %arg1: tensor<32x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func private @Unknown177(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    return %0 : tensor<1000x512xf32>
  }
  func private @Unknown178(%arg0: tensor<32x1000xf16>) -> tensor<32x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    return %0 : tensor<32x1000xf32>
  }
  func private @Unknown179(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000xf32>) -> tensor<1000xf16>
    %1 = "mhlo.convert"(%0) : (tensor<1000xf16>) -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func private @Unknown180(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown181(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown182(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown183(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown184(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown185(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown186(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown187(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown188(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown189(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown190(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown191(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown192(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown193(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown194(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown195(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown196(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown197(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown198(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown199(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown200(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown201(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown202(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown203(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown204(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown205(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown206(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown207(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown208(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown209(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown210(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown211(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown212(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown213(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown214(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown215(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown216(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown217(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown218(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown219(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown220(%arg0: tensor<1000xf32>) -> tensor<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000xf32>) -> tensor<1000xf16>
    return %0 : tensor<1000xf16>
  }
  func private @Unknown221(%arg0: tensor<1000xf16>, %arg1: tensor<32x1000xf16>) -> tensor<32x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<32x1000xf16>
    %1 = mhlo.add %arg1, %0 : tensor<32x1000xf16>
    return %1 : tensor<32x1000xf16>
  }
  func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<32x3x224x224xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64x64x3x3xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64x64x3x3xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<128x64x3x3xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128x64x1x1xf32>, %arg41: tensor<128x128x3x3xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x128x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x3x3xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256x256x3x3xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256x128x1x1xf32>, %arg66: tensor<256x256x3x3xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256x256x3x3xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<512x256x3x3xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512x256x1x1xf32>, %arg91: tensor<512x512x3x3xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512x512x3x3xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<1000x512xf32>, %arg102: tensor<32x1000xf16>, %arg103: tensor<1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>) {
    %0 = mhlo.constant dense<0xFC00> : tensor<f16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = mhlo.constant dense<1> : tensor<i64>
    %4 = call @Unknown0(%arg1) : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
    %5 = call @Unknown1(%arg0) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %6 = mhlo.convolution(%4, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16>
    %7:3 = call @BatchNormTrainingOp2(%6, %arg5, %arg4) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %8 = call @Unknown3(%arg101) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %9 = "mhlo.dot"(%arg102, %8) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x1000xf16>, tensor<1000x512xf16>) -> tensor<32x512xf16>
    %10 = call @Unknown4(%arg6) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %11 = call @Unknown5(%arg11) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %12 = call @Unknown6(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %13 = call @Unknown7(%arg21) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %14 = call @Unknown8(%arg26) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %15 = call @Unknown9(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %16 = call @Unknown10(%arg40) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %17 = call @Unknown11(%arg41) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %18 = call @Unknown12(%arg46) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %19 = call @Unknown13(%arg51) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %20 = call @Unknown14(%arg56) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %21 = call @Unknown15(%arg65) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %22 = call @Unknown16(%arg66) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %23 = call @Unknown17(%arg71) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %24 = call @Unknown18(%arg76) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %25 = call @Unknown19(%arg81) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %26 = call @Unknown20(%arg90) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %27 = call @Unknown21(%arg91) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %28 = call @Unknown22(%arg96) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %29:2 = call @Unknown23(%7#0) : (tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<32x64x112x112xi1>)
    %30 = "mhlo.reduce_window"(%29#0, %0) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%252) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
    %31 = mhlo.convolution(%30, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %32:3 = call @BatchNormTrainingOp24(%31, %arg10, %arg9) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %33:2 = call @Unknown25(%32#0) : (tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>)
    %34 = mhlo.convolution(%33#0, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %35:3 = call @BatchNormTrainingOp26(%34, %arg15, %arg14) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %36:2 = call @Unknown27(%35#0, %30) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>)
    %37 = mhlo.convolution(%36#0, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %38:3 = call @BatchNormTrainingOp28(%37, %arg20, %arg19) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %39:2 = call @Unknown29(%38#0) : (tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>)
    %40 = mhlo.convolution(%39#0, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %41:3 = call @BatchNormTrainingOp30(%40, %arg25, %arg24) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %42:2 = call @Unknown31(%41#0, %36#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>)
    %43 = mhlo.convolution(%42#0, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16>
    %44:3 = call @BatchNormTrainingOp32(%43, %arg30, %arg29) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %45 = mhlo.convolution(%42#0, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16>
    %46:3 = call @BatchNormTrainingOp33(%45, %arg39, %arg38) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %47:2 = call @Unknown34(%44#0) : (tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>)
    %48 = mhlo.convolution(%47#0, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %49:3 = call @BatchNormTrainingOp35(%48, %arg35, %arg34) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %50:2 = call @Unknown36(%49#0, %46#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>)
    %51 = mhlo.convolution(%50#0, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %52:3 = call @BatchNormTrainingOp37(%51, %arg45, %arg44) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %53:2 = call @Unknown38(%52#0) : (tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>)
    %54 = mhlo.convolution(%53#0, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %55:3 = call @BatchNormTrainingOp39(%54, %arg50, %arg49) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %56:2 = call @Unknown40(%55#0, %50#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>)
    %57 = mhlo.convolution(%56#0, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16>
    %58:3 = call @BatchNormTrainingOp41(%57, %arg55, %arg54) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %59 = mhlo.convolution(%56#0, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16>
    %60:3 = call @BatchNormTrainingOp42(%59, %arg64, %arg63) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %61:2 = call @Unknown43(%58#0) : (tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>)
    %62 = mhlo.convolution(%61#0, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %63:3 = call @BatchNormTrainingOp44(%62, %arg60, %arg59) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %64:2 = call @Unknown45(%63#0, %60#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>)
    %65 = mhlo.convolution(%64#0, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %66:3 = call @BatchNormTrainingOp46(%65, %arg70, %arg69) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %67:2 = call @Unknown47(%66#0) : (tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>)
    %68 = mhlo.convolution(%67#0, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %69:3 = call @BatchNormTrainingOp48(%68, %arg75, %arg74) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %70:2 = call @Unknown49(%69#0, %64#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>)
    %71 = mhlo.convolution(%70#0, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16>
    %72:3 = call @BatchNormTrainingOp50(%71, %arg80, %arg79) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %73 = mhlo.convolution(%70#0, %26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16>
    %74:3 = call @BatchNormTrainingOp51(%73, %arg89, %arg88) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %75:2 = call @Unknown52(%72#0) : (tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>)
    %76 = mhlo.convolution(%75#0, %25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %77:3 = call @BatchNormTrainingOp53(%76, %arg85, %arg84) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %78:2 = call @Unknown54(%77#0, %74#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>)
    %79 = mhlo.convolution(%78#0, %27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %80:3 = call @BatchNormTrainingOp55(%79, %arg95, %arg94) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %81:2 = call @Unknown56(%80#0) : (tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>)
    %82 = mhlo.convolution(%81#0, %28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %83:3 = call @BatchNormTrainingOp57(%82, %arg100, %arg99) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %84:2 = call @Unknown58(%9, %83#0, %78#0) : (tensor<32x512xf16>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>)
    %85 = call @Unknown59(%83#2) : (tensor<512xf32>) -> tensor<512xf32>
    %86:3 = call @BatchNormGradOp60(%82, %arg100, %83#1, %85, %84#1) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %87 = call @ConvBackwardDataOp61(%86#0, %28) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %88 = call @ConvBackwardFilterOp62(%81#0, %86#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %89 = call @Unknown63(%81#1, %87) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %90 = call @Unknown64(%80#2) : (tensor<512xf32>) -> tensor<512xf32>
    %91:3 = call @BatchNormGradOp65(%79, %arg95, %80#1, %90, %89) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %92 = call @ConvBackwardDataOp66(%91#0, %27) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %93 = call @ConvBackwardFilterOp67(%78#0, %91#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %94 = call @Unknown68(%84#1, %92, %78#1) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>) -> tensor<32x512x7x7xf16>
    %95 = call @Unknown69(%77#2) : (tensor<512xf32>) -> tensor<512xf32>
    %96:3 = call @BatchNormGradOp70(%76, %arg85, %77#1, %95, %94) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %97 = call @ConvBackwardDataOp71(%96#0, %25) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %98 = call @ConvBackwardFilterOp72(%75#0, %96#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %99 = call @Unknown73(%75#1, %97) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %100 = call @Unknown74(%72#2) : (tensor<512xf32>) -> tensor<512xf32>
    %101:3 = call @BatchNormGradOp75(%71, %arg80, %72#1, %100, %99) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %102 = call @ConvBackwardDataOp76(%101#0, %24) : (tensor<32x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %103 = call @ConvBackwardFilterOp77(%70#0, %101#0) : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %104 = call @Unknown78(%74#2) : (tensor<512xf32>) -> tensor<512xf32>
    %105:3 = call @BatchNormGradOp79(%73, %arg89, %74#1, %104, %94) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %106 = call @ConvBackwardDataOp80(%105#0, %26) : (tensor<32x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<32x256x14x14xf16>
    %107 = call @ConvBackwardFilterOp81(%70#0, %105#0) : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %108 = call @Unknown82(%106, %102, %70#1) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16>
    %109 = call @Unknown83(%69#2) : (tensor<256xf32>) -> tensor<256xf32>
    %110:3 = call @BatchNormGradOp84(%68, %arg75, %69#1, %109, %108) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %111 = call @ConvBackwardDataOp85(%110#0, %23) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %112 = call @ConvBackwardFilterOp86(%67#0, %110#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %113 = call @Unknown87(%67#1, %111) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %114 = call @Unknown88(%66#2) : (tensor<256xf32>) -> tensor<256xf32>
    %115:3 = call @BatchNormGradOp89(%65, %arg70, %66#1, %114, %113) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %116 = call @ConvBackwardDataOp90(%115#0, %22) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %117 = call @ConvBackwardFilterOp91(%64#0, %115#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %118 = call @Unknown92(%108, %116, %64#1) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16>
    %119 = call @Unknown93(%63#2) : (tensor<256xf32>) -> tensor<256xf32>
    %120:3 = call @BatchNormGradOp94(%62, %arg60, %63#1, %119, %118) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %121 = call @ConvBackwardDataOp95(%120#0, %20) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %122 = call @ConvBackwardFilterOp96(%61#0, %120#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %123 = call @Unknown97(%61#1, %121) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %124 = call @Unknown98(%58#2) : (tensor<256xf32>) -> tensor<256xf32>
    %125:3 = call @BatchNormGradOp99(%57, %arg55, %58#1, %124, %123) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %126 = call @ConvBackwardDataOp100(%125#0, %19) : (tensor<32x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %127 = call @ConvBackwardFilterOp101(%56#0, %125#0) : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %128 = call @Unknown102(%60#2) : (tensor<256xf32>) -> tensor<256xf32>
    %129:3 = call @BatchNormGradOp103(%59, %arg64, %60#1, %128, %118) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %130 = call @ConvBackwardDataOp104(%129#0, %21) : (tensor<32x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<32x128x28x28xf16>
    %131 = call @ConvBackwardFilterOp105(%56#0, %129#0) : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %132 = call @Unknown106(%130, %126, %56#1) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16>
    %133 = call @Unknown107(%55#2) : (tensor<128xf32>) -> tensor<128xf32>
    %134:3 = call @BatchNormGradOp108(%54, %arg50, %55#1, %133, %132) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %135 = call @ConvBackwardDataOp109(%134#0, %18) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %136 = call @ConvBackwardFilterOp110(%53#0, %134#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %137 = call @Unknown111(%53#1, %135) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %138 = call @Unknown112(%52#2) : (tensor<128xf32>) -> tensor<128xf32>
    %139:3 = call @BatchNormGradOp113(%51, %arg45, %52#1, %138, %137) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %140 = call @ConvBackwardDataOp114(%139#0, %17) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %141 = call @ConvBackwardFilterOp115(%50#0, %139#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %142 = call @Unknown116(%132, %140, %50#1) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16>
    %143 = call @Unknown117(%49#2) : (tensor<128xf32>) -> tensor<128xf32>
    %144:3 = call @BatchNormGradOp118(%48, %arg35, %49#1, %143, %142) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %145 = call @ConvBackwardDataOp119(%144#0, %15) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %146 = call @ConvBackwardFilterOp120(%47#0, %144#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %147 = call @Unknown121(%47#1, %145) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %148 = call @Unknown122(%44#2) : (tensor<128xf32>) -> tensor<128xf32>
    %149:3 = call @BatchNormGradOp123(%43, %arg30, %44#1, %148, %147) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %150 = call @ConvBackwardDataOp124(%149#0, %14) : (tensor<32x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %151 = call @ConvBackwardFilterOp125(%42#0, %149#0) : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %152 = call @Unknown126(%46#2) : (tensor<128xf32>) -> tensor<128xf32>
    %153:3 = call @BatchNormGradOp127(%45, %arg39, %46#1, %152, %142) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %154 = call @ConvBackwardDataOp128(%153#0, %16) : (tensor<32x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<32x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp129(%42#0, %153#0) : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %156 = call @Unknown130(%154, %150, %42#1) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16>
    %157 = call @Unknown131(%41#2) : (tensor<64xf32>) -> tensor<64xf32>
    %158:3 = call @BatchNormGradOp132(%40, %arg25, %41#1, %157, %156) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %159 = call @ConvBackwardDataOp133(%158#0, %13) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %160 = call @ConvBackwardFilterOp134(%39#0, %158#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %161 = call @Unknown135(%39#1, %159) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %162 = call @Unknown136(%38#2) : (tensor<64xf32>) -> tensor<64xf32>
    %163:3 = call @BatchNormGradOp137(%37, %arg20, %38#1, %162, %161) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %164 = call @ConvBackwardDataOp138(%163#0, %12) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %165 = call @ConvBackwardFilterOp139(%36#0, %163#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %166 = call @Unknown140(%156, %164, %36#1) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16>
    %167 = call @Unknown141(%35#2) : (tensor<64xf32>) -> tensor<64xf32>
    %168:3 = call @BatchNormGradOp142(%34, %arg15, %35#1, %167, %166) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %169 = call @ConvBackwardDataOp143(%168#0, %11) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %170 = call @ConvBackwardFilterOp144(%33#0, %168#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %171 = call @Unknown145(%33#1, %169) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %172 = call @Unknown146(%32#2) : (tensor<64xf32>) -> tensor<64xf32>
    %173:3 = call @BatchNormGradOp147(%31, %arg10, %32#1, %172, %171) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %174 = call @ConvBackwardDataOp148(%173#0, %10) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %175 = call @ConvBackwardFilterOp149(%30, %173#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %176 = call @Unknown150(%166, %174) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %177 = "mhlo.select_and_scatter"(%29#0, %176, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%252) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%252) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
    %178 = call @Unknown151(%29#1, %177) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %179 = call @Unknown152(%7#2) : (tensor<64xf32>) -> tensor<64xf32>
    %180:3 = call @BatchNormGradOp153(%6, %arg5, %7#1, %179, %178) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %181 = call @ConvBackwardFilterOp154(%4, %180#0) : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %182 = call @Unknown155(%181) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %183 = call @Unknown156(%175) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %184 = call @Unknown157(%170) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %185 = call @Unknown158(%165) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %186 = call @Unknown159(%160) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %187 = call @Unknown160(%151) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %188 = call @Unknown161(%146) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %189 = call @Unknown162(%155) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %190 = call @Unknown163(%141) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %191 = call @Unknown164(%136) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %192 = call @Unknown165(%127) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %193 = call @Unknown166(%122) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %194 = call @Unknown167(%131) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %195 = call @Unknown168(%117) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %196 = call @Unknown169(%112) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %197 = call @Unknown170(%103) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %198 = call @Unknown171(%98) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %199 = call @Unknown172(%107) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %200 = call @Unknown173(%93) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %201 = call @Unknown174(%88) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %202 = mhlo.reduce(%84#0 init: %1) across dimensions = [3, 2] : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %252 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%252) : (tensor<f16>) -> ()
    }
    %203 = call @Unknown175(%202) : (tensor<32x512xf16>) -> tensor<32x512xf16>
    %204 = call @MatmulOp176(%203, %arg102) : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<1000x512xf16>
    %205 = call @Unknown177(%204) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %206 = call @Unknown178(%arg102) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    %207 = mhlo.reduce(%206 init: %2) across dimensions = [0] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %252 = mhlo.add %arg104, %arg105 : tensor<f32>
      "mhlo.return"(%252) : (tensor<f32>) -> ()
    }
    %208 = call @Unknown179(%207) : (tensor<1000xf32>) -> tensor<1000xf32>
    %209 = call @Unknown180(%7#1, %arg3) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %210 = call @Unknown181(%7#2, %arg2) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %211 = call @Unknown182(%32#1, %arg8) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %212 = call @Unknown183(%32#2, %arg7) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %213 = call @Unknown184(%35#1, %arg13) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %214 = call @Unknown185(%35#2, %arg12) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %215 = call @Unknown186(%38#1, %arg18) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %216 = call @Unknown187(%38#2, %arg17) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %217 = call @Unknown188(%41#1, %arg23) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %218 = call @Unknown189(%41#2, %arg22) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %219 = call @Unknown190(%44#1, %arg28) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %220 = call @Unknown191(%44#2, %arg27) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %221 = call @Unknown192(%49#1, %arg33) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %222 = call @Unknown193(%49#2, %arg32) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %223 = call @Unknown194(%46#1, %arg37) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %224 = call @Unknown195(%46#2, %arg36) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %225 = call @Unknown196(%52#1, %arg43) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %226 = call @Unknown197(%52#2, %arg42) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %227 = call @Unknown198(%55#1, %arg48) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %228 = call @Unknown199(%55#2, %arg47) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %229 = call @Unknown200(%58#1, %arg53) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %230 = call @Unknown201(%58#2, %arg52) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %231 = call @Unknown202(%63#1, %arg58) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %232 = call @Unknown203(%63#2, %arg57) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %233 = call @Unknown204(%60#1, %arg62) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %234 = call @Unknown205(%60#2, %arg61) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %235 = call @Unknown206(%66#1, %arg68) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %236 = call @Unknown207(%66#2, %arg67) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %237 = call @Unknown208(%69#1, %arg73) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %238 = call @Unknown209(%69#2, %arg72) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %239 = call @Unknown210(%72#1, %arg78) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %240 = call @Unknown211(%72#2, %arg77) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %241 = call @Unknown212(%77#1, %arg83) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %242 = call @Unknown213(%77#2, %arg82) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %243 = call @Unknown214(%74#1, %arg87) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %244 = call @Unknown215(%74#2, %arg86) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %245 = call @Unknown216(%80#1, %arg93) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %246 = call @Unknown217(%80#2, %arg92) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %247 = call @Unknown218(%83#1, %arg98) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %248 = call @Unknown219(%83#2, %arg97) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %249 = call @Unknown220(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %250 = "mhlo.dot_general"(%203, %8) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<1000x512xf16>) -> tensor<32x1000xf16>
    %251 = call @Unknown221(%249, %250) : (tensor<1000xf16>, tensor<32x1000xf16>) -> tensor<32x1000xf16>
    return %182, %180#1, %180#2, %183, %173#1, %173#2, %184, %168#1, %168#2, %185, %163#1, %163#2, %186, %158#1, %158#2, %187, %149#1, %149#2, %188, %144#1, %144#2, %189, %153#1, %153#2, %190, %139#1, %139#2, %191, %134#1, %134#2, %192, %125#1, %125#2, %193, %120#1, %120#2, %194, %129#1, %129#2, %195, %115#1, %115#2, %196, %110#1, %110#2, %197, %101#1, %101#2, %198, %96#1, %96#2, %199, %105#1, %105#2, %200, %91#1, %91#2, %201, %86#1, %86#2, %205, %208, %209, %210, %3, %211, %212, %3, %213, %214, %3, %215, %216, %3, %217, %218, %3, %219, %220, %3, %221, %222, %3, %223, %224, %3, %225, %226, %3, %227, %228, %3, %229, %230, %3, %231, %232, %3, %233, %234, %3, %235, %236, %3, %237, %238, %3, %239, %240, %3, %241, %242, %3, %243, %244, %3, %245, %246, %3, %247, %248, %3, %251 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>
  }
}

