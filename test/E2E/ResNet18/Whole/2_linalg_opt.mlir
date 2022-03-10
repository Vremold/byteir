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
  func private @BatchNormGradOp59(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %4, %3#1, %3#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp60(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp61(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown62(%arg0: tensor<32x512x7x7xi1>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %1 : tensor<32x512x7x7xf16>
  }
  func private @BatchNormGradOp63(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %4, %3#1, %3#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp64(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp65(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown66(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>, %arg2: tensor<32x512x7x7xi1>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x512x7x7xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @BatchNormGradOp67(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %4, %3#1, %3#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp68(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp69(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown70(%arg0: tensor<32x512x7x7xi1>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %1 : tensor<32x512x7x7xf16>
  }
  func private @BatchNormGradOp71(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %4, %3#1, %3#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp72(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp73(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func private @BatchNormGradOp74(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<512xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    return %4, %3#1, %3#2 : tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp75(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp76(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func private @Unknown77(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @BatchNormGradOp78(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %4, %3#1, %3#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp79(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp80(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown81(%arg0: tensor<32x256x14x14xi1>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @BatchNormGradOp82(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %4, %3#1, %3#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp83(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp84(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown85(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @BatchNormGradOp86(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %4, %3#1, %3#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp87(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp88(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown89(%arg0: tensor<32x256x14x14xi1>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %1 : tensor<32x256x14x14xf16>
  }
  func private @BatchNormGradOp90(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %4, %3#1, %3#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp91(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp92(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func private @BatchNormGradOp93(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %4, %3#1, %3#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp94(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp95(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func private @Unknown96(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @BatchNormGradOp97(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %4, %3#1, %3#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp98(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp99(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown100(%arg0: tensor<32x128x28x28xi1>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @BatchNormGradOp101(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %4, %3#1, %3#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp102(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp103(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown104(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @BatchNormGradOp105(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %4, %3#1, %3#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp106(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp107(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown108(%arg0: tensor<32x128x28x28xi1>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %1 : tensor<32x128x28x28xf16>
  }
  func private @BatchNormGradOp109(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %4, %3#1, %3#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func private @BatchNormGradOp112(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    return %4, %3#1, %3#2 : tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp113(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp114(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func private @Unknown115(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @BatchNormGradOp116(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %4, %3#1, %3#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp117(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp118(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown119(%arg0: tensor<32x64x56x56xi1>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @BatchNormGradOp120(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %4, %3#1, %3#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp121(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp122(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown123(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    %2 = "mhlo.select"(%arg2, %1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @BatchNormGradOp124(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %4, %3#1, %3#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp125(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp126(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown127(%arg0: tensor<32x64x56x56xi1>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %1 : tensor<32x64x56x56xf16>
  }
  func private @BatchNormGradOp128(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    return %4, %3#1, %3#2 : tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp129(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp130(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown131(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    return %0 : tensor<32x64x56x56xf16>
  }
  func private @Unknown132(%arg0: tensor<32x64x112x112xi1>, %arg1: tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<32x64x112x112xf16>
    %1 = "mhlo.select"(%arg0, %arg1, %0) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    return %1 : tensor<32x64x112x112xf16>
  }
  func private @BatchNormGradOp133(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %1 = "mhlo.convert"(%arg2) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<64xf32>
    %3:3 = "mhlo.batch_norm_grad"(%0, %arg1, %2, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %4 = "mhlo.convert"(%3#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    return %4, %3#1, %3#2 : tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardFilterOp134(%arg0: tensor<32x3x224x224xf16>, %arg1: tensor<32x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func private @Unknown135(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    return %0 : tensor<64x3x7x7xf32>
  }
  func private @Unknown136(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown137(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown138(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown139(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown140(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    return %0 : tensor<128x64x3x3xf32>
  }
  func private @Unknown141(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown142(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    return %0 : tensor<128x64x1x1xf32>
  }
  func private @Unknown143(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown144(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown145(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    return %0 : tensor<256x128x3x3xf32>
  }
  func private @Unknown146(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown147(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    return %0 : tensor<256x128x1x1xf32>
  }
  func private @Unknown148(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown149(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown150(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    return %0 : tensor<512x256x3x3xf32>
  }
  func private @Unknown151(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown152(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    return %0 : tensor<512x256x1x1xf32>
  }
  func private @Unknown153(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown154(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown155(%arg0: tensor<32x512xf16>) -> tensor<32x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<2.040100e-02> : tensor<32x512xf16>
    %1 = mhlo.multiply %arg0, %0 : tensor<32x512xf16>
    return %1 : tensor<32x512xf16>
  }
  func private @MatmulOp156(%arg0: tensor<32x512xf16>, %arg1: tensor<32x1000xf16>) -> tensor<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %1 : tensor<1000x512xf16>
  }
  func private @Unknown157(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    return %0 : tensor<1000x512xf32>
  }
  func private @Unknown158(%arg0: tensor<32x1000xf16>) -> tensor<32x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    return %0 : tensor<32x1000xf32>
  }
  func private @Unknown159(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000xf32>) -> tensor<1000xf16>
    %1 = "mhlo.convert"(%0) : (tensor<1000xf16>) -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func private @Unknown160(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown161(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown162(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown163(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown164(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown165(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown166(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown167(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown168(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown169(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<64xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<64xf32>
    %4 = mhlo.add %2, %3 : tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func private @Unknown170(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown171(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown172(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown173(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown174(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown175(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown176(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown177(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown178(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown179(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<128xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<128xf32>
    %4 = mhlo.add %2, %3 : tensor<128xf32>
    return %4 : tensor<128xf32>
  }
  func private @Unknown180(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown181(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown182(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown183(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown184(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown185(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown186(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown187(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown188(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown189(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<256xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<256xf32>
    %4 = mhlo.add %2, %3 : tensor<256xf32>
    return %4 : tensor<256xf32>
  }
  func private @Unknown190(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown191(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown192(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown193(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown194(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown195(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown196(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown197(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown198(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown199(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.multiply %arg0, %1 : tensor<512xf32>
    %3 = mhlo.multiply %arg1, %0 : tensor<512xf32>
    %4 = mhlo.add %2, %3 : tensor<512xf32>
    return %4 : tensor<512xf32>
  }
  func private @Unknown200(%arg0: tensor<1000xf32>) -> tensor<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000xf32>) -> tensor<1000xf16>
    return %0 : tensor<1000xf16>
  }
  func private @Unknown201(%arg0: tensor<1000xf16>, %arg1: tensor<32x1000xf16>) -> tensor<32x1000xf16> attributes {__byteir_elementwise_fusion__} {
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
      %232 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%232) : (tensor<f16>) -> ()
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
    %85:3 = call @BatchNormGradOp59(%82, %arg100, %84#1) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %86 = call @ConvBackwardDataOp60(%85#0, %28) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %87 = call @ConvBackwardFilterOp61(%81#0, %85#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %88 = call @Unknown62(%81#1, %86) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %89:3 = call @BatchNormGradOp63(%79, %arg95, %88) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %90 = call @ConvBackwardDataOp64(%89#0, %27) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %91 = call @ConvBackwardFilterOp65(%78#0, %89#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %92 = call @Unknown66(%84#1, %90, %78#1) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xi1>) -> tensor<32x512x7x7xf16>
    %93:3 = call @BatchNormGradOp67(%76, %arg85, %92) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %94 = call @ConvBackwardDataOp68(%93#0, %25) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %95 = call @ConvBackwardFilterOp69(%75#0, %93#0) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %96 = call @Unknown70(%75#1, %94) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %97:3 = call @BatchNormGradOp71(%71, %arg80, %96) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %98 = call @ConvBackwardDataOp72(%97#0, %24) : (tensor<32x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %99 = call @ConvBackwardFilterOp73(%70#0, %97#0) : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %100:3 = call @BatchNormGradOp74(%73, %arg89, %92) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<32x512x7x7xf16>) -> (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %101 = call @ConvBackwardDataOp75(%100#0, %26) : (tensor<32x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<32x256x14x14xf16>
    %102 = call @ConvBackwardFilterOp76(%70#0, %100#0) : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %103 = call @Unknown77(%101, %98, %70#1) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16>
    %104:3 = call @BatchNormGradOp78(%68, %arg75, %103) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %105 = call @ConvBackwardDataOp79(%104#0, %23) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %106 = call @ConvBackwardFilterOp80(%67#0, %104#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %107 = call @Unknown81(%67#1, %105) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %108:3 = call @BatchNormGradOp82(%65, %arg70, %107) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %109 = call @ConvBackwardDataOp83(%108#0, %22) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %110 = call @ConvBackwardFilterOp84(%64#0, %108#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %111 = call @Unknown85(%103, %109, %64#1) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xi1>) -> tensor<32x256x14x14xf16>
    %112:3 = call @BatchNormGradOp86(%62, %arg60, %111) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %113 = call @ConvBackwardDataOp87(%112#0, %20) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %114 = call @ConvBackwardFilterOp88(%61#0, %112#0) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %115 = call @Unknown89(%61#1, %113) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %116:3 = call @BatchNormGradOp90(%57, %arg55, %115) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %117 = call @ConvBackwardDataOp91(%116#0, %19) : (tensor<32x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %118 = call @ConvBackwardFilterOp92(%56#0, %116#0) : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %119:3 = call @BatchNormGradOp93(%59, %arg64, %111) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<32x256x14x14xf16>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %120 = call @ConvBackwardDataOp94(%119#0, %21) : (tensor<32x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<32x128x28x28xf16>
    %121 = call @ConvBackwardFilterOp95(%56#0, %119#0) : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %122 = call @Unknown96(%120, %117, %56#1) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16>
    %123:3 = call @BatchNormGradOp97(%54, %arg50, %122) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %124 = call @ConvBackwardDataOp98(%123#0, %18) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %125 = call @ConvBackwardFilterOp99(%53#0, %123#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %126 = call @Unknown100(%53#1, %124) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %127:3 = call @BatchNormGradOp101(%51, %arg45, %126) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %128 = call @ConvBackwardDataOp102(%127#0, %17) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %129 = call @ConvBackwardFilterOp103(%50#0, %127#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %130 = call @Unknown104(%122, %128, %50#1) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xi1>) -> tensor<32x128x28x28xf16>
    %131:3 = call @BatchNormGradOp105(%48, %arg35, %130) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %132 = call @ConvBackwardDataOp106(%131#0, %15) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %133 = call @ConvBackwardFilterOp107(%47#0, %131#0) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %134 = call @Unknown108(%47#1, %132) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %135:3 = call @BatchNormGradOp109(%43, %arg30, %134) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %136 = call @ConvBackwardDataOp110(%135#0, %14) : (tensor<32x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %137 = call @ConvBackwardFilterOp111(%42#0, %135#0) : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %138:3 = call @BatchNormGradOp112(%45, %arg39, %130) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<32x128x28x28xf16>) -> (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %139 = call @ConvBackwardDataOp113(%138#0, %16) : (tensor<32x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<32x64x56x56xf16>
    %140 = call @ConvBackwardFilterOp114(%42#0, %138#0) : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %141 = call @Unknown115(%139, %136, %42#1) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16>
    %142:3 = call @BatchNormGradOp116(%40, %arg25, %141) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %143 = call @ConvBackwardDataOp117(%142#0, %13) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %144 = call @ConvBackwardFilterOp118(%39#0, %142#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %145 = call @Unknown119(%39#1, %143) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %146:3 = call @BatchNormGradOp120(%37, %arg20, %145) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %147 = call @ConvBackwardDataOp121(%146#0, %12) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %148 = call @ConvBackwardFilterOp122(%36#0, %146#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %149 = call @Unknown123(%141, %147, %36#1) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xi1>) -> tensor<32x64x56x56xf16>
    %150:3 = call @BatchNormGradOp124(%34, %arg15, %149) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %151 = call @ConvBackwardDataOp125(%150#0, %11) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %152 = call @ConvBackwardFilterOp126(%33#0, %150#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %153 = call @Unknown127(%33#1, %151) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %154:3 = call @BatchNormGradOp128(%31, %arg10, %153) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<32x64x56x56xf16>) -> (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %155 = call @ConvBackwardDataOp129(%154#0, %10) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %156 = call @ConvBackwardFilterOp130(%30, %154#0) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %157 = call @Unknown131(%149, %155) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %158 = "mhlo.select_and_scatter"(%29#0, %157, %1) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %232 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%232) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %232 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%232) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
    %159 = call @Unknown132(%29#1, %158) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %160:3 = call @BatchNormGradOp133(%6, %arg5, %159) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<32x64x112x112xf16>) -> (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %161 = call @ConvBackwardFilterOp134(%4, %160#0) : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %162 = call @Unknown135(%161) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %163 = call @Unknown136(%156) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %164 = call @Unknown137(%152) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %165 = call @Unknown138(%148) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %166 = call @Unknown139(%144) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %167 = call @Unknown140(%137) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %168 = call @Unknown141(%133) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %169 = call @Unknown142(%140) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %170 = call @Unknown143(%129) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %171 = call @Unknown144(%125) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %172 = call @Unknown145(%118) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %173 = call @Unknown146(%114) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %174 = call @Unknown147(%121) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %175 = call @Unknown148(%110) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %176 = call @Unknown149(%106) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %177 = call @Unknown150(%99) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %178 = call @Unknown151(%95) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %179 = call @Unknown152(%102) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %180 = call @Unknown153(%91) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %181 = call @Unknown154(%87) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %182 = mhlo.reduce(%84#0 init: %1) across dimensions = [3, 2] : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %232 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%232) : (tensor<f16>) -> ()
    }
    %183 = call @Unknown155(%182) : (tensor<32x512xf16>) -> tensor<32x512xf16>
    %184 = call @MatmulOp156(%183, %arg102) : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<1000x512xf16>
    %185 = call @Unknown157(%184) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %186 = call @Unknown158(%arg102) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    %187 = mhlo.reduce(%186 init: %2) across dimensions = [0] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %232 = mhlo.add %arg104, %arg105 : tensor<f32>
      "mhlo.return"(%232) : (tensor<f32>) -> ()
    }
    %188 = call @Unknown159(%187) : (tensor<1000xf32>) -> tensor<1000xf32>
    %189 = call @Unknown160(%7#1, %arg3) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %190 = call @Unknown161(%7#2, %arg2) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %191 = call @Unknown162(%32#1, %arg8) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %192 = call @Unknown163(%32#2, %arg7) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %193 = call @Unknown164(%35#1, %arg13) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %194 = call @Unknown165(%35#2, %arg12) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %195 = call @Unknown166(%38#1, %arg18) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %196 = call @Unknown167(%38#2, %arg17) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %197 = call @Unknown168(%41#1, %arg23) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %198 = call @Unknown169(%41#2, %arg22) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    %199 = call @Unknown170(%44#1, %arg28) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %200 = call @Unknown171(%44#2, %arg27) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %201 = call @Unknown172(%49#1, %arg33) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %202 = call @Unknown173(%49#2, %arg32) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %203 = call @Unknown174(%46#1, %arg37) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %204 = call @Unknown175(%46#2, %arg36) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %205 = call @Unknown176(%52#1, %arg43) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %206 = call @Unknown177(%52#2, %arg42) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %207 = call @Unknown178(%55#1, %arg48) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %208 = call @Unknown179(%55#2, %arg47) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %209 = call @Unknown180(%58#1, %arg53) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %210 = call @Unknown181(%58#2, %arg52) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %211 = call @Unknown182(%63#1, %arg58) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %212 = call @Unknown183(%63#2, %arg57) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %213 = call @Unknown184(%60#1, %arg62) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %214 = call @Unknown185(%60#2, %arg61) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %215 = call @Unknown186(%66#1, %arg68) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %216 = call @Unknown187(%66#2, %arg67) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %217 = call @Unknown188(%69#1, %arg73) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %218 = call @Unknown189(%69#2, %arg72) : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %219 = call @Unknown190(%72#1, %arg78) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %220 = call @Unknown191(%72#2, %arg77) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %221 = call @Unknown192(%77#1, %arg83) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %222 = call @Unknown193(%77#2, %arg82) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %223 = call @Unknown194(%74#1, %arg87) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %224 = call @Unknown195(%74#2, %arg86) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %225 = call @Unknown196(%80#1, %arg93) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %226 = call @Unknown197(%80#2, %arg92) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %227 = call @Unknown198(%83#1, %arg98) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %228 = call @Unknown199(%83#2, %arg97) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    %229 = call @Unknown200(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %230 = "mhlo.dot_general"(%183, %8) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<1000x512xf16>) -> tensor<32x1000xf16>
    %231 = call @Unknown201(%229, %230) : (tensor<1000xf16>, tensor<32x1000xf16>) -> tensor<32x1000xf16>
    return %162, %160#1, %160#2, %163, %154#1, %154#2, %164, %150#1, %150#2, %165, %146#1, %146#2, %166, %142#1, %142#2, %167, %135#1, %135#2, %168, %131#1, %131#2, %169, %138#1, %138#2, %170, %127#1, %127#2, %171, %123#1, %123#2, %172, %116#1, %116#2, %173, %112#1, %112#2, %174, %119#1, %119#2, %175, %108#1, %108#2, %176, %104#1, %104#2, %177, %97#1, %97#2, %178, %93#1, %93#2, %179, %100#1, %100#2, %180, %89#1, %89#2, %181, %85#1, %85#2, %185, %188, %189, %190, %3, %191, %192, %3, %193, %194, %3, %195, %196, %3, %197, %198, %3, %199, %200, %3, %201, %202, %3, %203, %204, %3, %205, %206, %3, %207, %208, %3, %209, %210, %3, %211, %212, %3, %213, %214, %3, %215, %216, %3, %217, %218, %3, %219, %220, %3, %221, %222, %3, %223, %224, %3, %225, %226, %3, %227, %228, %3, %231 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>
  }
}

