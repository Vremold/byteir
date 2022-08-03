// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func.func @main
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
module {
  func.func private @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.900000e+01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : memref<1x512x7x7xf16>, memref<1x512xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.divf %arg3, %cst : f16
      %2 = arith.cmpf ogt, %arg2, %cst_0 : f16
      %3 = arith.select %2, %1, %cst_0 : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp1(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp2(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %2 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp3(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp5(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp6(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %2 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp7(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp9(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp10(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %2 : memref<1x512x7x7xf16>
  }
  func.func private @ConvBackwardFilterOp11(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func.func private @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @BatchNormGradOp13(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp14(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<3x3x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %2 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp15(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func.func private @BatchNormGradOp16(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @ConvBackwardDataOp17(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<1x1x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp18(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func.func private @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp20(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp21(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %2 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp22(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp24(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp25(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %2 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp26(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp28(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp29(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %2 : memref<1x256x14x14xf16>
  }
  func.func private @ConvBackwardFilterOp30(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func.func private @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @BatchNormGradOp32(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<3x3x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %2 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp34(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func.func private @BatchNormGradOp35(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @ConvBackwardDataOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<1x1x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp37(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func.func private @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp39(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp40(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %2 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp41(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp43(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp44(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %2 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp45(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp47(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp48(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %2 : memref<1x128x28x28xf16>
  }
  func.func private @ConvBackwardFilterOp49(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func.func private @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @BatchNormGradOp51(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp52(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<3x3x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %2 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp53(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func.func private @BatchNormGradOp54(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @ConvBackwardDataOp55(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<1x1x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp56(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func.func private @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp58(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp59(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %2 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp60(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp62(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp63(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %2 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp64(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16):
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = arith.cmpf ogt, %arg3, %cst : f16
      %3 = arith.select %2, %1, %cst : f16
      linalg.yield %3 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp66(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp67(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %2 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp68(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @BatchNormGradOp70(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardDataOp71(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %2 : memref<1x64x56x56xf16>
  }
  func.func private @ConvBackwardFilterOp72(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func.func private @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x112x112xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) outs(%0 : memref<1x64x112x112xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.cmpf ogt, %arg2, %cst : f16
      %2 = arith.select %1, %arg3, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x64x112x112xf16>
  }
  func.func private @BatchNormGradOp75(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    %2 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    %3 = memref.alloc() : memref<1x64x112x112xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<1x64x112x112xf16>
    "lmhlo.convert"(%3, %6) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %6, %4, %5 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @ConvBackwardFilterOp76(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func.func private @Unknown77(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf16>) outs(%0 : memref<64x3x7x7xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x3x7x7xf32>
  }
  func.func private @Unknown78(%arg0: memref<1x1000xf16>) -> memref<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x1000xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x1000xf16>) outs(%0 : memref<1x1000xf32>) {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<1x1000xf32>
  }
  func.func private @Unknown79(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000xf32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg0 : memref<1000xf32>) outs(%0 : memref<1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.truncf %arg1 : f32 to f16
      %2 = arith.extf %1 : f16 to f32
      linalg.yield %2 : f32
    }
    return %0 : memref<1000xf32>
  }
  func.func private @Unknown80(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf16>) outs(%0 : memref<1000x512xf32>) {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<1000x512xf32>
  }
  func.func private @Unknown81(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func.func private @Unknown82(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func.func private @Unknown83(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func.func private @Unknown84(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func.func private @Unknown85(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf16>) outs(%0 : memref<128x64x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x64x3x3xf32>
  }
  func.func private @Unknown86(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func.func private @Unknown87(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf16>) outs(%0 : memref<128x64x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func.func private @Unknown88(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func.func private @Unknown89(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func.func private @Unknown90(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf16>) outs(%0 : memref<256x128x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x128x3x3xf32>
  }
  func.func private @Unknown91(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func.func private @Unknown92(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf16>) outs(%0 : memref<256x128x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func.func private @Unknown93(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func.func private @Unknown94(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func.func private @Unknown95(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf16>) outs(%0 : memref<512x256x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x256x3x3xf32>
  }
  func.func private @Unknown96(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func.func private @Unknown97(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf16>) outs(%0 : memref<512x256x1x1xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func.func private @Unknown98(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func.func private @Unknown99(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} {
    ^bb0(%arg1: f16, %arg2: f32):
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func.func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<128xf32>, %arg11: memref<128xf32>, %arg12: memref<128xf32>, %arg13: memref<128xf32>, %arg14: memref<128xf32>, %arg15: memref<128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<256xf32>, %arg21: memref<256xf32>, %arg22: memref<256xf32>, %arg23: memref<256xf32>, %arg24: memref<256xf32>, %arg25: memref<256xf32>, %arg26: memref<256xf32>, %arg27: memref<256xf32>, %arg28: memref<256xf32>, %arg29: memref<256xf32>, %arg30: memref<512xf32>, %arg31: memref<512xf32>, %arg32: memref<512xf32>, %arg33: memref<512xf32>, %arg34: memref<512xf32>, %arg35: memref<512xf32>, %arg36: memref<512xf32>, %arg37: memref<512xf32>, %arg38: memref<512xf32>, %arg39: memref<512xf32>, %arg40: memref<64xf32>, %arg41: memref<64xf32>, %arg42: memref<64xf32>, %arg43: memref<64xf32>, %arg44: memref<64xf32>, %arg45: memref<64xf32>, %arg46: memref<64xf32>, %arg47: memref<64xf32>, %arg48: memref<64xf32>, %arg49: memref<64xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<128xf32>, %arg53: memref<128xf32>, %arg54: memref<128xf32>, %arg55: memref<128xf32>, %arg56: memref<128xf32>, %arg57: memref<128xf32>, %arg58: memref<128xf32>, %arg59: memref<128xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<512xf32>, %arg71: memref<512xf32>, %arg72: memref<512xf32>, %arg73: memref<512xf32>, %arg74: memref<512xf32>, %arg75: memref<512xf32>, %arg76: memref<512xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<64x3x7x7xf16>, %arg81: memref<1x3x224x224xf16>, %arg82: memref<1x64x112x112xf16>, %arg83: memref<1x64x112x112xf16>, %arg84: memref<1x64x56x56xf16>, %arg85: memref<64x64x3x3xf16>, %arg86: memref<1x64x56x56xf16>, %arg87: memref<1x64x56x56xf16>, %arg88: memref<64x64x3x3xf16>, %arg89: memref<1x64x56x56xf16>, %arg90: memref<1x64x56x56xf16>, %arg91: memref<64x64x3x3xf16>, %arg92: memref<1x64x56x56xf16>, %arg93: memref<1x64x56x56xf16>, %arg94: memref<64x64x3x3xf16>, %arg95: memref<1x64x56x56xf16>, %arg96: memref<1x64x56x56xf16>, %arg97: memref<128x64x3x3xf16>, %arg98: memref<1x128x28x28xf16>, %arg99: memref<1x128x28x28xf16>, %arg100: memref<128x128x3x3xf16>, %arg101: memref<1x128x28x28xf16>, %arg102: memref<128x64x1x1xf16>, %arg103: memref<1x128x28x28xf16>, %arg104: memref<1x128x28x28xf16>, %arg105: memref<128x128x3x3xf16>, %arg106: memref<1x128x28x28xf16>, %arg107: memref<1x128x28x28xf16>, %arg108: memref<128x128x3x3xf16>, %arg109: memref<1x128x28x28xf16>, %arg110: memref<1x128x28x28xf16>, %arg111: memref<256x128x3x3xf16>, %arg112: memref<1x256x14x14xf16>, %arg113: memref<1x256x14x14xf16>, %arg114: memref<256x256x3x3xf16>, %arg115: memref<1x256x14x14xf16>, %arg116: memref<256x128x1x1xf16>, %arg117: memref<1x256x14x14xf16>, %arg118: memref<1x256x14x14xf16>, %arg119: memref<256x256x3x3xf16>, %arg120: memref<1x256x14x14xf16>, %arg121: memref<1x256x14x14xf16>, %arg122: memref<256x256x3x3xf16>, %arg123: memref<1x256x14x14xf16>, %arg124: memref<1x256x14x14xf16>, %arg125: memref<512x256x3x3xf16>, %arg126: memref<1x512x7x7xf16>, %arg127: memref<1x512x7x7xf16>, %arg128: memref<512x512x3x3xf16>, %arg129: memref<1x512x7x7xf16>, %arg130: memref<512x256x1x1xf16>, %arg131: memref<1x512x7x7xf16>, %arg132: memref<1x512x7x7xf16>, %arg133: memref<512x512x3x3xf16>, %arg134: memref<1x512x7x7xf16>, %arg135: memref<1x512x7x7xf16>, %arg136: memref<512x512x3x3xf16>, %arg137: memref<1x512x7x7xf16>, %arg138: memref<1x512x7x7xf16>, %arg139: memref<1x512xf16>, %arg140: memref<512x1000xf16>, %arg141: memref<1x1000xf16>) -> (memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>) {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    %2 = memref.alloc() : memref<1x512xf16>
    "lmhlo.dot"(%arg141, %arg140, %2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>) -> ()
    %3 = call @Unknown0(%2, %arg138) : (memref<1x512xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %4:3 = call @BatchNormGradOp1(%arg137, %arg39, %3) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %5 = call @ConvBackwardDataOp2(%4#0, %arg136) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %6 = call @ConvBackwardFilterOp3(%arg135, %4#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %7 = call @Unknown4(%arg135, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %8:3 = call @BatchNormGradOp5(%arg134, %arg37, %7) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %9 = call @ConvBackwardDataOp6(%8#0, %arg133) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %10 = call @ConvBackwardFilterOp7(%arg132, %8#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %11 = call @Unknown8(%3, %9, %arg132) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %12:3 = call @BatchNormGradOp9(%arg129, %arg33, %11) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %13 = call @ConvBackwardDataOp10(%12#0, %arg128) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %14 = call @ConvBackwardFilterOp11(%arg127, %12#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %15 = call @Unknown12(%arg127, %13) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %16:3 = call @BatchNormGradOp13(%arg126, %arg31, %15) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %17 = call @ConvBackwardDataOp14(%16#0, %arg125) : (memref<1x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %18 = call @ConvBackwardFilterOp15(%arg124, %16#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %19:3 = call @BatchNormGradOp16(%arg131, %arg35, %11) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %20 = call @ConvBackwardDataOp17(%19#0, %arg130) : (memref<1x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16>
    %21 = call @ConvBackwardFilterOp18(%arg124, %19#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %22 = call @Unknown19(%20, %17, %arg124) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %23:3 = call @BatchNormGradOp20(%arg123, %arg29, %22) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %24 = call @ConvBackwardDataOp21(%23#0, %arg122) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %25 = call @ConvBackwardFilterOp22(%arg121, %23#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %26 = call @Unknown23(%arg121, %24) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %27:3 = call @BatchNormGradOp24(%arg120, %arg27, %26) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %28 = call @ConvBackwardDataOp25(%27#0, %arg119) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %29 = call @ConvBackwardFilterOp26(%arg118, %27#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %30 = call @Unknown27(%22, %28, %arg118) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %31:3 = call @BatchNormGradOp28(%arg115, %arg23, %30) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %32 = call @ConvBackwardDataOp29(%31#0, %arg114) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %33 = call @ConvBackwardFilterOp30(%arg113, %31#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %34 = call @Unknown31(%arg113, %32) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %35:3 = call @BatchNormGradOp32(%arg112, %arg21, %34) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %36 = call @ConvBackwardDataOp33(%35#0, %arg111) : (memref<1x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %37 = call @ConvBackwardFilterOp34(%arg110, %35#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %38:3 = call @BatchNormGradOp35(%arg117, %arg25, %30) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %39 = call @ConvBackwardDataOp36(%38#0, %arg116) : (memref<1x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16>
    %40 = call @ConvBackwardFilterOp37(%arg110, %38#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %41 = call @Unknown38(%39, %36, %arg110) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %42:3 = call @BatchNormGradOp39(%arg109, %arg19, %41) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %43 = call @ConvBackwardDataOp40(%42#0, %arg108) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %44 = call @ConvBackwardFilterOp41(%arg107, %42#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %45 = call @Unknown42(%arg107, %43) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %46:3 = call @BatchNormGradOp43(%arg106, %arg17, %45) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %47 = call @ConvBackwardDataOp44(%46#0, %arg105) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %48 = call @ConvBackwardFilterOp45(%arg104, %46#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %49 = call @Unknown46(%41, %47, %arg104) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %50:3 = call @BatchNormGradOp47(%arg101, %arg13, %49) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %51 = call @ConvBackwardDataOp48(%50#0, %arg100) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %52 = call @ConvBackwardFilterOp49(%arg99, %50#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %53 = call @Unknown50(%arg99, %51) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %54:3 = call @BatchNormGradOp51(%arg98, %arg11, %53) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %55 = call @ConvBackwardDataOp52(%54#0, %arg97) : (memref<1x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %56 = call @ConvBackwardFilterOp53(%arg96, %54#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %57:3 = call @BatchNormGradOp54(%arg103, %arg15, %49) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %58 = call @ConvBackwardDataOp55(%57#0, %arg102) : (memref<1x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16>
    %59 = call @ConvBackwardFilterOp56(%arg96, %57#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %60 = call @Unknown57(%58, %55, %arg96) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %61:3 = call @BatchNormGradOp58(%arg95, %arg9, %60) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %62 = call @ConvBackwardDataOp59(%61#0, %arg94) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %63 = call @ConvBackwardFilterOp60(%arg93, %61#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %64 = call @Unknown61(%arg93, %62) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %65:3 = call @BatchNormGradOp62(%arg92, %arg7, %64) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %66 = call @ConvBackwardDataOp63(%65#0, %arg91) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %67 = call @ConvBackwardFilterOp64(%arg90, %65#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %68 = call @Unknown65(%60, %66, %arg90) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %69:3 = call @BatchNormGradOp66(%arg89, %arg5, %68) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %70 = call @ConvBackwardDataOp67(%69#0, %arg88) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %71 = call @ConvBackwardFilterOp68(%arg87, %69#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %72 = call @Unknown69(%arg87, %70) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %73:3 = call @BatchNormGradOp70(%arg86, %arg3, %72) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %74 = call @ConvBackwardDataOp71(%73#0, %arg85) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %75 = call @ConvBackwardFilterOp72(%arg84, %73#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %76 = call @Unknown73(%68, %74) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %77 = memref.alloc() : memref<1x64x112x112xf16>
    "lmhlo.select_and_scatter"(%arg83, %76, %1, %77) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %107 = "mhlo.compare"(%arg142, %arg143) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%107) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %107 = mhlo.add %arg142, %arg143 : tensor<f16>
      "mhlo.return"(%107) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<f16>, memref<1x64x112x112xf16>) -> ()
    %78 = call @Unknown74(%arg83, %77) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %79:3 = call @BatchNormGradOp75(%arg82, %arg1, %78) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %80 = call @ConvBackwardFilterOp76(%arg81, %79#0) : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %81 = call @Unknown77(%80) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %82 = call @Unknown78(%arg141) : (memref<1x1000xf16>) -> memref<1x1000xf32>
    %83 = memref.alloc() : memref<1000xf32>
    "lmhlo.reduce"(%82, %0, %83) ({
    ^bb0(%arg142: memref<f32>, %arg143: memref<f32>, %arg144: memref<f32>):
      "lmhlo.add"(%arg142, %arg143, %arg144) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<1x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %84 = call @Unknown79(%83) : (memref<1000xf32>) -> memref<1000xf32>
    %85 = memref.alloc() : memref<1000x1xf16>
    "lmhlo.reshape"(%arg141, %85) : (memref<1x1000xf16>, memref<1000x1xf16>) -> ()
    %86 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%85, %arg139, %86) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1000x1xf16>, memref<1x512xf16>, memref<1000x512xf16>) -> ()
    %87 = call @Unknown80(%86) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %88 = call @Unknown81(%75) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %89 = call @Unknown82(%71) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %90 = call @Unknown83(%67) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %91 = call @Unknown84(%63) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %92 = call @Unknown85(%56) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %93 = call @Unknown86(%52) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %94 = call @Unknown87(%59) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %95 = call @Unknown88(%48) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %96 = call @Unknown89(%44) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %97 = call @Unknown90(%37) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %98 = call @Unknown91(%33) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %99 = call @Unknown92(%40) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %100 = call @Unknown93(%29) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %101 = call @Unknown94(%25) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %102 = call @Unknown95(%18) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %103 = call @Unknown96(%14) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %104 = call @Unknown97(%21) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %105 = call @Unknown98(%10) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %106 = call @Unknown99(%6) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    return %79#2, %79#1, %81, %84, %87, %73#2, %73#1, %69#2, %69#1, %88, %89, %65#2, %65#1, %61#2, %61#1, %90, %91, %54#2, %54#1, %50#2, %50#1, %92, %93, %94, %57#2, %57#1, %46#2, %46#1, %42#2, %42#1, %95, %96, %35#2, %35#1, %31#2, %31#1, %97, %98, %99, %38#2, %38#1, %27#2, %27#1, %23#2, %23#1, %100, %101, %16#2, %16#1, %12#2, %12#1, %102, %103, %104, %19#2, %19#1, %8#2, %8#1, %4#2, %4#1, %105, %106 : memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>
  }
}

