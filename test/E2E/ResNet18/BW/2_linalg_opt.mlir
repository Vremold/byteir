// RUN: byteir-opt %s -linalg-opt | FileCheck %s

// CHECK-LABEL: func @main
module {
  func private @BatchNormTrainingOp0(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#1, %1#2 : tensor<64xf32>, tensor<64xf32>
  }
  func private @BatchNormTrainingOp1(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#1, %1#2 : tensor<64xf32>, tensor<64xf32>
  }
  func private @BatchNormTrainingOp2(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#1, %1#2 : tensor<64xf32>, tensor<64xf32>
  }
  func private @BatchNormTrainingOp3(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#1, %1#2 : tensor<64xf32>, tensor<64xf32>
  }
  func private @BatchNormTrainingOp4(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#1, %1#2 : tensor<64xf32>, tensor<64xf32>
  }
  func private @BatchNormTrainingOp5(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    return %1#1, %1#2 : tensor<128xf32>, tensor<128xf32>
  }
  func private @BatchNormTrainingOp6(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    return %1#1, %1#2 : tensor<128xf32>, tensor<128xf32>
  }
  func private @BatchNormTrainingOp7(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    return %1#1, %1#2 : tensor<128xf32>, tensor<128xf32>
  }
  func private @BatchNormTrainingOp8(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    return %1#1, %1#2 : tensor<128xf32>, tensor<128xf32>
  }
  func private @BatchNormTrainingOp9(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %1#1, %1#2 : tensor<256xf32>, tensor<256xf32>
  }
  func private @BatchNormTrainingOp10(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %1#1, %1#2 : tensor<256xf32>, tensor<256xf32>
  }
  func private @BatchNormTrainingOp11(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %1#1, %1#2 : tensor<256xf32>, tensor<256xf32>
  }
  func private @BatchNormTrainingOp12(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %1#1, %1#2 : tensor<256xf32>, tensor<256xf32>
  }
  func private @BatchNormTrainingOp13(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    return %1#1, %1#2 : tensor<512xf32>, tensor<512xf32>
  }
  func private @BatchNormTrainingOp14(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    return %1#1, %1#2 : tensor<512xf32>, tensor<512xf32>
  }
  func private @BatchNormTrainingOp15(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    return %1#1, %1#2 : tensor<512xf32>, tensor<512xf32>
  }
  func private @BatchNormTrainingOp16(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    return %1#1, %1#2 : tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown17(%arg0: tensor<1x512xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<4.900000e+01> : tensor<1x512x7x7xf16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x512xf16>) -> tensor<1x512x7x7xf16>
    %3 = mhlo.divide %2, %0 : tensor<1x512x7x7xf16>
    %4 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xi1>
    %5 = "mhlo.select"(%4, %3, %1) : (tensor<1x512x7x7xi1>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    return %5 : tensor<1x512x7x7xf16>
  }
  func private @Unknown18(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp19(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp20(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp21(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x512x7x7xi1>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func private @Unknown23(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp24(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp25(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp26(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown27(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>, %arg2: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x512x7x7xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x512x7x7xi1>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    return %3 : tensor<1x512x7x7xf16>
  }
  func private @Unknown28(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp29(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp30(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp31(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    return %1 : tensor<512x512x3x3xf16>
  }
  func private @Unknown32(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x512x7x7xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x512x7x7xi1>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    return %2 : tensor<1x512x7x7xf16>
  }
  func private @Unknown33(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp34(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp35(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp36(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    return %1 : tensor<512x256x3x3xf16>
  }
  func private @BatchNormTrainingOp37(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    return %1#1, %1#2 : tensor<512xf32>, tensor<512xf32>
  }
  func private @Unknown38(%arg0: tensor<512xf32>) -> tensor<512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %2 = mhlo.add %arg0, %0 : tensor<512xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<512xf32>) -> tensor<512xf32>
    %4 = mhlo.divide %1, %3 : tensor<512xf32>
    %5 = mhlo.multiply %4, %4 : tensor<512xf32>
    %6 = mhlo.subtract %5, %0 : tensor<512xf32>
    return %6 : tensor<512xf32>
  }
  func private @BatchNormGradOp39(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf32>) -> (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf16>
    return %3, %2#1, %2#2 : tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>
  }
  func private @ConvBackwardDataOp40(%arg0: tensor<1x512x7x7xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<1x256x14x14xf16>
    return %1 : tensor<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp41(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x512x7x7xf16>) -> tensor<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    return %1 : tensor<512x256x1x1xf16>
  }
  func private @Unknown42(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x256x14x14xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x256x14x14xi1>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    return %3 : tensor<1x256x14x14xf16>
  }
  func private @Unknown43(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp44(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp45(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp46(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown47(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x256x14x14xi1>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @Unknown48(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp49(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp50(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp51(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown52(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>, %arg2: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x256x14x14xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x256x14x14xi1>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    return %3 : tensor<1x256x14x14xf16>
  }
  func private @Unknown53(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp54(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp55(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp56(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    return %1 : tensor<256x256x3x3xf16>
  }
  func private @Unknown57(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x256x14x14xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x256x14x14xi1>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    return %2 : tensor<1x256x14x14xf16>
  }
  func private @Unknown58(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp59(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp60(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp61(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    return %1 : tensor<256x128x3x3xf16>
  }
  func private @BatchNormTrainingOp62(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %1#1, %1#2 : tensor<256xf32>, tensor<256xf32>
  }
  func private @Unknown63(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %2 = mhlo.add %arg0, %0 : tensor<256xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<256xf32>) -> tensor<256xf32>
    %4 = mhlo.divide %1, %3 : tensor<256xf32>
    %5 = mhlo.multiply %4, %4 : tensor<256xf32>
    %6 = mhlo.subtract %5, %0 : tensor<256xf32>
    return %6 : tensor<256xf32>
  }
  func private @BatchNormGradOp64(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf32>) -> (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf16>
    return %3, %2#1, %2#2 : tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
  }
  func private @ConvBackwardDataOp65(%arg0: tensor<1x256x14x14xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<1x128x28x28xf16>
    return %1 : tensor<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp66(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x256x14x14xf16>) -> tensor<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    return %1 : tensor<256x128x1x1xf16>
  }
  func private @Unknown67(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x128x28x28xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x128x28x28xi1>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    return %3 : tensor<1x128x28x28xf16>
  }
  func private @Unknown68(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp69(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp70(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp71(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown72(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x128x28x28xi1>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @Unknown73(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp74(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp75(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp76(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown77(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>, %arg2: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x128x28x28xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x128x28x28xi1>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    return %3 : tensor<1x128x28x28xf16>
  }
  func private @Unknown78(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp79(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    return %1 : tensor<128x128x3x3xf16>
  }
  func private @Unknown82(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x128x28x28xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x128x28x28xi1>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    return %2 : tensor<1x128x28x28xf16>
  }
  func private @Unknown83(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp84(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp85(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp86(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    return %1 : tensor<128x64x3x3xf16>
  }
  func private @BatchNormTrainingOp87(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    return %1#1, %1#2 : tensor<128xf32>, tensor<128xf32>
  }
  func private @Unknown88(%arg0: tensor<128xf32>) -> tensor<128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %2 = mhlo.add %arg0, %0 : tensor<128xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<128xf32>) -> tensor<128xf32>
    %4 = mhlo.divide %1, %3 : tensor<128xf32>
    %5 = mhlo.multiply %4, %4 : tensor<128xf32>
    %6 = mhlo.subtract %5, %0 : tensor<128xf32>
    return %6 : tensor<128xf32>
  }
  func private @BatchNormGradOp89(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf32>) -> (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf16>
    return %3, %2#1, %2#2 : tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>
  }
  func private @ConvBackwardDataOp90(%arg0: tensor<1x128x28x28xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp91(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x128x28x28xf16>) -> tensor<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    return %1 : tensor<128x64x1x1xf16>
  }
  func private @Unknown92(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x64x56x56xi1>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    return %3 : tensor<1x64x56x56xf16>
  }
  func private @Unknown93(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp94(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown97(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x64x56x56xi1>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @Unknown98(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp99(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp100(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp101(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown102(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>, %arg2: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    %2 = "mhlo.compare"(%arg2, %0) {comparison_direction = "GT"} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xi1>
    %3 = "mhlo.select"(%2, %1, %0) : (tensor<1x64x56x56xi1>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    return %3 : tensor<1x64x56x56xf16>
  }
  func private @Unknown103(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp104(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp105(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp106(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown107(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x56x56xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x64x56x56xi1>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @Unknown108(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp109(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %3, %2#1, %2#2 : tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = "mhlo.transpose"(%arg1) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    return %1 : tensor<64x64x3x3xf16>
  }
  func private @Unknown112(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<1x64x56x56xf16>
    return %0 : tensor<1x64x56x56xf16>
  }
  func private @Unknown113(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<1x64x112x112xf16>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "GT"} : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xi1>
    %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<1x64x112x112xi1>, tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    return %2 : tensor<1x64x112x112xf16>
  }
  func private @Unknown114(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %2 = mhlo.add %arg0, %0 : tensor<64xf32>
    %3 = "mhlo.rsqrt"(%2) : (tensor<64xf32>) -> tensor<64xf32>
    %4 = mhlo.divide %1, %3 : tensor<64xf32>
    %5 = mhlo.multiply %4, %4 : tensor<64xf32>
    %6 = mhlo.subtract %5, %0 : tensor<64xf32>
    return %6 : tensor<64xf32>
  }
  func private @BatchNormGradOp115(%arg0: tensor<1x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<1x64x112x112xf16>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %1 = "mhlo.convert"(%arg4) : (tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf32>
    %2:3 = "mhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x112x112xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%2#0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf16>
    return %3, %2#1, %2#2 : tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func private @ConvBackwardFilterOp116(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<1x64x112x112xf16>) -> tensor<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    return %1 : tensor<64x3x7x7xf16>
  }
  func private @Unknown117(%arg0: tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    return %0 : tensor<64x3x7x7xf32>
  }
  func private @Unknown118(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    return %0 : tensor<1x1000xf32>
  }
  func private @Unknown119(%arg0: tensor<1000xf32>) -> tensor<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000xf32>) -> tensor<1000xf16>
    %1 = "mhlo.convert"(%0) : (tensor<1000xf16>) -> tensor<1000xf32>
    return %1 : tensor<1000xf32>
  }
  func private @Unknown120(%arg0: tensor<1000x512xf16>) -> tensor<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    return %0 : tensor<1000x512xf32>
  }
  func private @Unknown121(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown122(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown123(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown124(%arg0: tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    return %0 : tensor<64x64x3x3xf32>
  }
  func private @Unknown125(%arg0: tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    return %0 : tensor<128x64x3x3xf32>
  }
  func private @Unknown126(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown127(%arg0: tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    return %0 : tensor<128x64x1x1xf32>
  }
  func private @Unknown128(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown129(%arg0: tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    return %0 : tensor<128x128x3x3xf32>
  }
  func private @Unknown130(%arg0: tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    return %0 : tensor<256x128x3x3xf32>
  }
  func private @Unknown131(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown132(%arg0: tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    return %0 : tensor<256x128x1x1xf32>
  }
  func private @Unknown133(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown134(%arg0: tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    return %0 : tensor<256x256x3x3xf32>
  }
  func private @Unknown135(%arg0: tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    return %0 : tensor<512x256x3x3xf32>
  }
  func private @Unknown136(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown137(%arg0: tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    return %0 : tensor<512x256x1x1xf32>
  }
  func private @Unknown138(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func private @Unknown139(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.convert"(%arg0) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %0 : tensor<512x512x3x3xf32>
  }
  func @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<256xf32>, %arg21: tensor<256xf32>, %arg22: tensor<256xf32>, %arg23: tensor<256xf32>, %arg24: tensor<256xf32>, %arg25: tensor<256xf32>, %arg26: tensor<256xf32>, %arg27: tensor<256xf32>, %arg28: tensor<256xf32>, %arg29: tensor<256xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<64xf32>, %arg41: tensor<64xf32>, %arg42: tensor<64xf32>, %arg43: tensor<64xf32>, %arg44: tensor<64xf32>, %arg45: tensor<64xf32>, %arg46: tensor<64xf32>, %arg47: tensor<64xf32>, %arg48: tensor<64xf32>, %arg49: tensor<64xf32>, %arg50: tensor<128xf32>, %arg51: tensor<128xf32>, %arg52: tensor<128xf32>, %arg53: tensor<128xf32>, %arg54: tensor<128xf32>, %arg55: tensor<128xf32>, %arg56: tensor<128xf32>, %arg57: tensor<128xf32>, %arg58: tensor<128xf32>, %arg59: tensor<128xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256xf32>, %arg66: tensor<256xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<512xf32>, %arg71: tensor<512xf32>, %arg72: tensor<512xf32>, %arg73: tensor<512xf32>, %arg74: tensor<512xf32>, %arg75: tensor<512xf32>, %arg76: tensor<512xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<64x3x7x7xf16>, %arg81: tensor<1x3x224x224xf16>, %arg82: tensor<1x64x112x112xf16>, %arg83: tensor<1x64x112x112xf16>, %arg84: tensor<1x64x56x56xf16>, %arg85: tensor<64x64x3x3xf16>, %arg86: tensor<1x64x56x56xf16>, %arg87: tensor<1x64x56x56xf16>, %arg88: tensor<64x64x3x3xf16>, %arg89: tensor<1x64x56x56xf16>, %arg90: tensor<1x64x56x56xf16>, %arg91: tensor<64x64x3x3xf16>, %arg92: tensor<1x64x56x56xf16>, %arg93: tensor<1x64x56x56xf16>, %arg94: tensor<64x64x3x3xf16>, %arg95: tensor<1x64x56x56xf16>, %arg96: tensor<1x64x56x56xf16>, %arg97: tensor<128x64x3x3xf16>, %arg98: tensor<1x128x28x28xf16>, %arg99: tensor<1x128x28x28xf16>, %arg100: tensor<128x128x3x3xf16>, %arg101: tensor<1x128x28x28xf16>, %arg102: tensor<128x64x1x1xf16>, %arg103: tensor<1x128x28x28xf16>, %arg104: tensor<1x128x28x28xf16>, %arg105: tensor<128x128x3x3xf16>, %arg106: tensor<1x128x28x28xf16>, %arg107: tensor<1x128x28x28xf16>, %arg108: tensor<128x128x3x3xf16>, %arg109: tensor<1x128x28x28xf16>, %arg110: tensor<1x128x28x28xf16>, %arg111: tensor<256x128x3x3xf16>, %arg112: tensor<1x256x14x14xf16>, %arg113: tensor<1x256x14x14xf16>, %arg114: tensor<256x256x3x3xf16>, %arg115: tensor<1x256x14x14xf16>, %arg116: tensor<256x128x1x1xf16>, %arg117: tensor<1x256x14x14xf16>, %arg118: tensor<1x256x14x14xf16>, %arg119: tensor<256x256x3x3xf16>, %arg120: tensor<1x256x14x14xf16>, %arg121: tensor<1x256x14x14xf16>, %arg122: tensor<256x256x3x3xf16>, %arg123: tensor<1x256x14x14xf16>, %arg124: tensor<1x256x14x14xf16>, %arg125: tensor<512x256x3x3xf16>, %arg126: tensor<1x512x7x7xf16>, %arg127: tensor<1x512x7x7xf16>, %arg128: tensor<512x512x3x3xf16>, %arg129: tensor<1x512x7x7xf16>, %arg130: tensor<512x256x1x1xf16>, %arg131: tensor<1x512x7x7xf16>, %arg132: tensor<1x512x7x7xf16>, %arg133: tensor<512x512x3x3xf16>, %arg134: tensor<1x512x7x7xf16>, %arg135: tensor<1x512x7x7xf16>, %arg136: tensor<512x512x3x3xf16>, %arg137: tensor<1x512x7x7xf16>, %arg138: tensor<1x512x7x7xf16>, %arg139: tensor<1x512xf16>, %arg140: tensor<512x1000xf16>, %arg141: tensor<1x1000xf16>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2:2 = call @BatchNormTrainingOp0(%arg82, %arg1, %arg0) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    %3:2 = call @BatchNormTrainingOp1(%arg86, %arg3, %arg2) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    %4:2 = call @BatchNormTrainingOp2(%arg89, %arg5, %arg4) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    %5:2 = call @BatchNormTrainingOp3(%arg92, %arg7, %arg6) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    %6:2 = call @BatchNormTrainingOp4(%arg95, %arg9, %arg8) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
    %7:2 = call @BatchNormTrainingOp5(%arg98, %arg11, %arg10) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
    %8:2 = call @BatchNormTrainingOp6(%arg101, %arg13, %arg12) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
    %9:2 = call @BatchNormTrainingOp7(%arg106, %arg17, %arg16) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
    %10:2 = call @BatchNormTrainingOp8(%arg109, %arg19, %arg18) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
    %11:2 = call @BatchNormTrainingOp9(%arg112, %arg21, %arg20) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %12:2 = call @BatchNormTrainingOp10(%arg115, %arg23, %arg22) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %13:2 = call @BatchNormTrainingOp11(%arg120, %arg27, %arg26) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %14:2 = call @BatchNormTrainingOp12(%arg123, %arg29, %arg28) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %15:2 = call @BatchNormTrainingOp13(%arg126, %arg31, %arg30) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>)
    %16:2 = call @BatchNormTrainingOp14(%arg129, %arg33, %arg32) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>)
    %17:2 = call @BatchNormTrainingOp15(%arg134, %arg37, %arg36) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>)
    %18:2 = call @BatchNormTrainingOp16(%arg137, %arg39, %arg38) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>)
    %19 = "mhlo.dot_general"(%arg141, %arg140) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1000xf16>, tensor<512x1000xf16>) -> tensor<1x512xf16>
    %20 = call @Unknown17(%19, %arg138) : (tensor<1x512xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %21 = call @Unknown18(%18#1) : (tensor<512xf32>) -> tensor<512xf32>
    %22:3 = call @BatchNormGradOp19(%arg137, %arg39, %18#0, %21, %20) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %23 = call @ConvBackwardDataOp20(%22#0, %arg136) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %24 = call @ConvBackwardFilterOp21(%arg135, %22#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %25 = call @Unknown22(%arg135, %23) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %26 = call @Unknown23(%17#1) : (tensor<512xf32>) -> tensor<512xf32>
    %27:3 = call @BatchNormGradOp24(%arg134, %arg37, %17#0, %26, %25) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %28 = call @ConvBackwardDataOp25(%27#0, %arg133) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %29 = call @ConvBackwardFilterOp26(%arg132, %27#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %30 = call @Unknown27(%20, %28, %arg132) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %31 = call @Unknown28(%16#1) : (tensor<512xf32>) -> tensor<512xf32>
    %32:3 = call @BatchNormGradOp29(%arg129, %arg33, %16#0, %31, %30) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %33 = call @ConvBackwardDataOp30(%32#0, %arg128) : (tensor<1x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<1x512x7x7xf16>
    %34 = call @ConvBackwardFilterOp31(%arg127, %32#0) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<512x512x3x3xf16>
    %35 = call @Unknown32(%arg127, %33) : (tensor<1x512x7x7xf16>, tensor<1x512x7x7xf16>) -> tensor<1x512x7x7xf16>
    %36 = call @Unknown33(%15#1) : (tensor<512xf32>) -> tensor<512xf32>
    %37:3 = call @BatchNormGradOp34(%arg126, %arg31, %15#0, %36, %35) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %38 = call @ConvBackwardDataOp35(%37#0, %arg125) : (tensor<1x512x7x7xf16>, tensor<512x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %39 = call @ConvBackwardFilterOp36(%arg124, %37#0) : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<512x256x3x3xf16>
    %40:2 = call @BatchNormTrainingOp37(%arg131, %arg35, %arg34) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> (tensor<512xf32>, tensor<512xf32>)
    %41 = call @Unknown38(%40#1) : (tensor<512xf32>) -> tensor<512xf32>
    %42:3 = call @BatchNormGradOp39(%arg131, %arg35, %40#0, %41, %30) : (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1x512x7x7xf16>) -> (tensor<1x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>)
    %43 = call @ConvBackwardDataOp40(%42#0, %arg130) : (tensor<1x512x7x7xf16>, tensor<512x256x1x1xf16>) -> tensor<1x256x14x14xf16>
    %44 = call @ConvBackwardFilterOp41(%arg124, %42#0) : (tensor<1x256x14x14xf16>, tensor<1x512x7x7xf16>) -> tensor<512x256x1x1xf16>
    %45 = call @Unknown42(%43, %38, %arg124) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %46 = call @Unknown43(%14#1) : (tensor<256xf32>) -> tensor<256xf32>
    %47:3 = call @BatchNormGradOp44(%arg123, %arg29, %14#0, %46, %45) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %48 = call @ConvBackwardDataOp45(%47#0, %arg122) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %49 = call @ConvBackwardFilterOp46(%arg121, %47#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %50 = call @Unknown47(%arg121, %48) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %51 = call @Unknown48(%13#1) : (tensor<256xf32>) -> tensor<256xf32>
    %52:3 = call @BatchNormGradOp49(%arg120, %arg27, %13#0, %51, %50) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %53 = call @ConvBackwardDataOp50(%52#0, %arg119) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %54 = call @ConvBackwardFilterOp51(%arg118, %52#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %55 = call @Unknown52(%45, %53, %arg118) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %56 = call @Unknown53(%12#1) : (tensor<256xf32>) -> tensor<256xf32>
    %57:3 = call @BatchNormGradOp54(%arg115, %arg23, %12#0, %56, %55) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %58 = call @ConvBackwardDataOp55(%57#0, %arg114) : (tensor<1x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<1x256x14x14xf16>
    %59 = call @ConvBackwardFilterOp56(%arg113, %57#0) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<256x256x3x3xf16>
    %60 = call @Unknown57(%arg113, %58) : (tensor<1x256x14x14xf16>, tensor<1x256x14x14xf16>) -> tensor<1x256x14x14xf16>
    %61 = call @Unknown58(%11#1) : (tensor<256xf32>) -> tensor<256xf32>
    %62:3 = call @BatchNormGradOp59(%arg112, %arg21, %11#0, %61, %60) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %63 = call @ConvBackwardDataOp60(%62#0, %arg111) : (tensor<1x256x14x14xf16>, tensor<256x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %64 = call @ConvBackwardFilterOp61(%arg110, %62#0) : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<256x128x3x3xf16>
    %65:2 = call @BatchNormTrainingOp62(%arg117, %arg25, %arg24) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> (tensor<256xf32>, tensor<256xf32>)
    %66 = call @Unknown63(%65#1) : (tensor<256xf32>) -> tensor<256xf32>
    %67:3 = call @BatchNormGradOp64(%arg117, %arg25, %65#0, %66, %55) : (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1x256x14x14xf16>) -> (tensor<1x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>)
    %68 = call @ConvBackwardDataOp65(%67#0, %arg116) : (tensor<1x256x14x14xf16>, tensor<256x128x1x1xf16>) -> tensor<1x128x28x28xf16>
    %69 = call @ConvBackwardFilterOp66(%arg110, %67#0) : (tensor<1x128x28x28xf16>, tensor<1x256x14x14xf16>) -> tensor<256x128x1x1xf16>
    %70 = call @Unknown67(%68, %63, %arg110) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %71 = call @Unknown68(%10#1) : (tensor<128xf32>) -> tensor<128xf32>
    %72:3 = call @BatchNormGradOp69(%arg109, %arg19, %10#0, %71, %70) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %73 = call @ConvBackwardDataOp70(%72#0, %arg108) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %74 = call @ConvBackwardFilterOp71(%arg107, %72#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %75 = call @Unknown72(%arg107, %73) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %76 = call @Unknown73(%9#1) : (tensor<128xf32>) -> tensor<128xf32>
    %77:3 = call @BatchNormGradOp74(%arg106, %arg17, %9#0, %76, %75) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %78 = call @ConvBackwardDataOp75(%77#0, %arg105) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %79 = call @ConvBackwardFilterOp76(%arg104, %77#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %80 = call @Unknown77(%70, %78, %arg104) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %81 = call @Unknown78(%8#1) : (tensor<128xf32>) -> tensor<128xf32>
    %82:3 = call @BatchNormGradOp79(%arg101, %arg13, %8#0, %81, %80) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %83 = call @ConvBackwardDataOp80(%82#0, %arg100) : (tensor<1x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<1x128x28x28xf16>
    %84 = call @ConvBackwardFilterOp81(%arg99, %82#0) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<128x128x3x3xf16>
    %85 = call @Unknown82(%arg99, %83) : (tensor<1x128x28x28xf16>, tensor<1x128x28x28xf16>) -> tensor<1x128x28x28xf16>
    %86 = call @Unknown83(%7#1) : (tensor<128xf32>) -> tensor<128xf32>
    %87:3 = call @BatchNormGradOp84(%arg98, %arg11, %7#0, %86, %85) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %88 = call @ConvBackwardDataOp85(%87#0, %arg97) : (tensor<1x128x28x28xf16>, tensor<128x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %89 = call @ConvBackwardFilterOp86(%arg96, %87#0) : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<128x64x3x3xf16>
    %90:2 = call @BatchNormTrainingOp87(%arg103, %arg15, %arg14) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>)
    %91 = call @Unknown88(%90#1) : (tensor<128xf32>) -> tensor<128xf32>
    %92:3 = call @BatchNormGradOp89(%arg103, %arg15, %90#0, %91, %80) : (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x28x28xf16>) -> (tensor<1x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>)
    %93 = call @ConvBackwardDataOp90(%92#0, %arg102) : (tensor<1x128x28x28xf16>, tensor<128x64x1x1xf16>) -> tensor<1x64x56x56xf16>
    %94 = call @ConvBackwardFilterOp91(%arg96, %92#0) : (tensor<1x64x56x56xf16>, tensor<1x128x28x28xf16>) -> tensor<128x64x1x1xf16>
    %95 = call @Unknown92(%93, %88, %arg96) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %96 = call @Unknown93(%6#1) : (tensor<64xf32>) -> tensor<64xf32>
    %97:3 = call @BatchNormGradOp94(%arg95, %arg9, %6#0, %96, %95) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %98 = call @ConvBackwardDataOp95(%97#0, %arg94) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %99 = call @ConvBackwardFilterOp96(%arg93, %97#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %100 = call @Unknown97(%arg93, %98) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %101 = call @Unknown98(%5#1) : (tensor<64xf32>) -> tensor<64xf32>
    %102:3 = call @BatchNormGradOp99(%arg92, %arg7, %5#0, %101, %100) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %103 = call @ConvBackwardDataOp100(%102#0, %arg91) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %104 = call @ConvBackwardFilterOp101(%arg90, %102#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %105 = call @Unknown102(%95, %103, %arg90) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %106 = call @Unknown103(%4#1) : (tensor<64xf32>) -> tensor<64xf32>
    %107:3 = call @BatchNormGradOp104(%arg89, %arg5, %4#0, %106, %105) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %108 = call @ConvBackwardDataOp105(%107#0, %arg88) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %109 = call @ConvBackwardFilterOp106(%arg87, %107#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %110 = call @Unknown107(%arg87, %108) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %111 = call @Unknown108(%3#1) : (tensor<64xf32>) -> tensor<64xf32>
    %112:3 = call @BatchNormGradOp109(%arg86, %arg3, %3#0, %111, %110) : (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x56x56xf16>) -> (tensor<1x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>)
    %113 = call @ConvBackwardDataOp110(%112#0, %arg85) : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
    %114 = call @ConvBackwardFilterOp111(%arg84, %112#0) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<64x64x3x3xf16>
    %115 = call @Unknown112(%105, %113) : (tensor<1x64x56x56xf16>, tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf16>
    %116 = "mhlo.select_and_scatter"(%arg83, %115, %0) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):  // no predecessors
      %146 = "mhlo.compare"(%arg142, %arg143) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%146) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):  // no predecessors
      %146 = mhlo.add %arg142, %arg143 : tensor<f16>
      "mhlo.return"(%146) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<f16>) -> tensor<1x64x112x112xf16>
    %117 = call @Unknown113(%arg83, %116) : (tensor<1x64x112x112xf16>, tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    %118 = call @Unknown114(%2#1) : (tensor<64xf32>) -> tensor<64xf32>
    %119:3 = call @BatchNormGradOp115(%arg82, %arg1, %2#0, %118, %117) : (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1x64x112x112xf16>) -> (tensor<1x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>)
    %120 = call @ConvBackwardFilterOp116(%arg81, %119#0) : (tensor<1x3x224x224xf16>, tensor<1x64x112x112xf16>) -> tensor<64x3x7x7xf16>
    %121 = call @Unknown117(%120) : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %122 = call @Unknown118(%arg141) : (tensor<1x1000xf16>) -> tensor<1x1000xf32>
    %123 = mhlo.reduce(%122 init: %1) across dimensions = [0] : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg142: tensor<f32>, %arg143: tensor<f32>)  {
      %146 = mhlo.add %arg142, %arg143 : tensor<f32>
      "mhlo.return"(%146) : (tensor<f32>) -> ()
    }
    %124 = call @Unknown119(%123) : (tensor<1000xf32>) -> tensor<1000xf32>
    %125 = "mhlo.dot_general"(%arg141, %arg139) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1000xf16>, tensor<1x512xf16>) -> tensor<1000x512xf16>
    %126 = call @Unknown120(%125) : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %127 = call @Unknown121(%114) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %128 = call @Unknown122(%109) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %129 = call @Unknown123(%104) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %130 = call @Unknown124(%99) : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %131 = call @Unknown125(%89) : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %132 = call @Unknown126(%84) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %133 = call @Unknown127(%94) : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %134 = call @Unknown128(%79) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %135 = call @Unknown129(%74) : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %136 = call @Unknown130(%64) : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %137 = call @Unknown131(%59) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %138 = call @Unknown132(%69) : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %139 = call @Unknown133(%54) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %140 = call @Unknown134(%49) : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %141 = call @Unknown135(%39) : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %142 = call @Unknown136(%34) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %143 = call @Unknown137(%44) : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %144 = call @Unknown138(%29) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %145 = call @Unknown139(%24) : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    return %119#2, %119#1, %121, %124, %126, %112#2, %112#1, %107#2, %107#1, %127, %128, %102#2, %102#1, %97#2, %97#1, %129, %130, %87#2, %87#1, %82#2, %82#1, %131, %132, %133, %92#2, %92#1, %77#2, %77#1, %72#2, %72#1, %134, %135, %62#2, %62#1, %57#2, %57#1, %136, %137, %138, %67#2, %67#1, %52#2, %52#1, %47#2, %47#1, %139, %140, %37#2, %37#1, %32#2, %32#1, %141, %142, %143, %42#2, %42#1, %27#2, %27#1, %22#2, %22#1, %144, %145 : tensor<64xf32>, tensor<64xf32>, tensor<64x3x7x7xf32>, tensor<1000xf32>, tensor<1000x512xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128x128x3x3xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256x256x3x3xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512x512x3x3xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512x512x3x3xf32>
  }
}

