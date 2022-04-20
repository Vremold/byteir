// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0) -> (d0 mod 7)>
#map1 = affine_map<(d0) -> ((d0 floordiv 7) mod 7)>
#map2 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 512)>
#map3 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 512)>
#map4 = affine_map<(d0) -> (d0 mod 14)>
#map5 = affine_map<(d0) -> ((d0 floordiv 14) mod 14)>
#map6 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) mod 256)>
#map7 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) floordiv 256)>
#map8 = affine_map<(d0) -> (d0 mod 28)>
#map9 = affine_map<(d0) -> ((d0 floordiv 28) mod 28)>
#map10 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) mod 128)>
#map11 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) floordiv 128)>
#map12 = affine_map<(d0) -> (d0 mod 56)>
#map13 = affine_map<(d0) -> ((d0 floordiv 56) mod 56)>
#map14 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) mod 64)>
#map15 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) floordiv 64)>
#map16 = affine_map<(d0) -> (d0 mod 112)>
#map17 = affine_map<(d0) -> ((d0 floordiv 112) mod 112)>
#map18 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) mod 64)>
#map19 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) floordiv 64)>
#map20 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 3)>
#map21 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 3)>
#map22 = affine_map<(d0) -> (d0 mod 1000)>
#map23 = affine_map<(d0) -> (d0 floordiv 1000)>
#map24 = affine_map<(d0) -> (d0 mod 512)>
#map25 = affine_map<(d0) -> (d0 floordiv 512)>
#map26 = affine_map<(d0) -> (d0 mod 3)>
#map27 = affine_map<(d0) -> ((d0 floordiv 3) mod 3)>
#map28 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 64)>
#map29 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 64)>
#map30 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 128)>
#map31 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 128)>
#map32 = affine_map<(d0) -> (d0 mod 64)>
#map33 = affine_map<(d0) -> (d0 floordiv 64)>
#map34 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 256)>
#map35 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 256)>
#map36 = affine_map<(d0) -> (d0 mod 128)>
#map37 = affine_map<(d0) -> (d0 floordiv 128)>
#map38 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 512)>
#map39 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 512)>
#map40 = affine_map<(d0) -> (d0 mod 256)>
#map41 = affine_map<(d0) -> (d0 floordiv 256)>
module {
  func private @Unknown0(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    %cst_0 = arith.constant 4.900000e+01 : f16
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.apply #map2(%arg2)
      %4 = affine.apply #map3(%arg2)
      %5 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg0[%4, %3] : memref<1x512xf16>
      %7 = arith.divf %6, %cst_0 : f16
      %8 = arith.cmpf ogt, %5, %cst : f16
      %9 = arith.select %8, %7, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @BatchNormGradOp1(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    %6 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp2(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp3(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown4(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.apply #map2(%arg2)
      %4 = affine.apply #map3(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @BatchNormGradOp5(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    %6 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp6(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp7(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown8(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    affine.for %arg3 = 0 to 25088 {
      %1 = affine.apply #map0(%arg3)
      %2 = affine.apply #map1(%arg3)
      %3 = affine.apply #map2(%arg3)
      %4 = affine.apply #map3(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @BatchNormGradOp9(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    %6 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp10(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp11(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown12(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.apply #map2(%arg2)
      %4 = affine.apply #map3(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @BatchNormGradOp13(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    %6 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp14(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp15(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @BatchNormGradOp16(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    %6 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp17(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x1x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp18(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown19(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    affine.for %arg3 = 0 to 50176 {
      %1 = affine.apply #map4(%arg3)
      %2 = affine.apply #map5(%arg3)
      %3 = affine.apply #map6(%arg3)
      %4 = affine.apply #map7(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @BatchNormGradOp20(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    %6 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp21(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp22(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown23(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map6(%arg2)
      %4 = affine.apply #map7(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @BatchNormGradOp24(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    %6 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp25(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp26(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown27(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    affine.for %arg3 = 0 to 50176 {
      %1 = affine.apply #map4(%arg3)
      %2 = affine.apply #map5(%arg3)
      %3 = affine.apply #map6(%arg3)
      %4 = affine.apply #map7(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @BatchNormGradOp28(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    %6 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp29(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp30(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map6(%arg2)
      %4 = affine.apply #map7(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @BatchNormGradOp32(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    %6 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp34(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @BatchNormGradOp35(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    %6 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x1x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp37(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown38(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    affine.for %arg3 = 0 to 100352 {
      %1 = affine.apply #map8(%arg3)
      %2 = affine.apply #map9(%arg3)
      %3 = affine.apply #map10(%arg3)
      %4 = affine.apply #map11(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @BatchNormGradOp39(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    %6 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp40(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp41(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown42(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map8(%arg2)
      %2 = affine.apply #map9(%arg2)
      %3 = affine.apply #map10(%arg2)
      %4 = affine.apply #map11(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @BatchNormGradOp43(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    %6 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp44(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp45(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown46(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    affine.for %arg3 = 0 to 100352 {
      %1 = affine.apply #map8(%arg3)
      %2 = affine.apply #map9(%arg3)
      %3 = affine.apply #map10(%arg3)
      %4 = affine.apply #map11(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @BatchNormGradOp47(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    %6 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp48(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp49(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown50(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map8(%arg2)
      %2 = affine.apply #map9(%arg2)
      %3 = affine.apply #map10(%arg2)
      %4 = affine.apply #map11(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @BatchNormGradOp51(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    %6 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp52(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp53(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @BatchNormGradOp54(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    %6 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp55(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x1x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp56(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown57(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    affine.for %arg3 = 0 to 200704 {
      %1 = affine.apply #map12(%arg3)
      %2 = affine.apply #map13(%arg3)
      %3 = affine.apply #map14(%arg3)
      %4 = affine.apply #map15(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @BatchNormGradOp58(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    %6 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp59(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp60(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown61(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map12(%arg2)
      %2 = affine.apply #map13(%arg2)
      %3 = affine.apply #map14(%arg2)
      %4 = affine.apply #map15(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @BatchNormGradOp62(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    %6 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp63(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp64(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown65(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    affine.for %arg3 = 0 to 200704 {
      %1 = affine.apply #map12(%arg3)
      %2 = affine.apply #map13(%arg3)
      %3 = affine.apply #map14(%arg3)
      %4 = affine.apply #map15(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.cmpf ogt, %5, %cst : f16
      %10 = arith.select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @BatchNormGradOp66(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    %6 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp67(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp68(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown69(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map12(%arg2)
      %2 = affine.apply #map13(%arg2)
      %3 = affine.apply #map14(%arg2)
      %4 = affine.apply #map15(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @BatchNormGradOp70(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    %6 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp71(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp72(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown73(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map12(%arg2)
      %2 = affine.apply #map13(%arg2)
      %3 = affine.apply #map14(%arg2)
      %4 = affine.apply #map15(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.addf %5, %6 : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown74(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x112x112xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map16(%arg2)
      %2 = affine.apply #map17(%arg2)
      %3 = affine.apply #map18(%arg2)
      %4 = affine.apply #map19(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x112x112xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = arith.select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
    }
    return %0 : memref<1x64x112x112xf16>
  }
  func private @BatchNormGradOp75(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<1x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x112x112xf32>
    %5 = memref.alloc() : memref<1x64x112x112xf32>
    %6 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp76(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown77(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf32>
    affine.for %arg1 = 0 to 9408 {
      %1 = affine.apply #map0(%arg1)
      %2 = affine.apply #map1(%arg1)
      %3 = affine.apply #map20(%arg1)
      %4 = affine.apply #map21(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x3x7x7xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x3x7x7xf32>
    }
    return %0 : memref<64x3x7x7xf32>
  }
  func private @Unknown78(%arg0: memref<1x1000xf16>) -> memref<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.apply #map22(%arg1)
      %2 = affine.apply #map23(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1x1000xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1x1000xf32>
    }
    return %0 : memref<1x1000xf32>
  }
  func private @Unknown79(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      affine.store %3, %0[%arg1] : memref<1000xf32>
    }
    return %0 : memref<1000xf32>
  }
  func private @Unknown80(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf32>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map24(%arg1)
      %2 = affine.apply #map25(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1000x512xf32>
    }
    return %0 : memref<1000x512xf32>
  }
  func private @Unknown81(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map28(%arg1)
      %4 = affine.apply #map29(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown82(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map28(%arg1)
      %4 = affine.apply #map29(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown83(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map28(%arg1)
      %4 = affine.apply #map29(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown84(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map28(%arg1)
      %4 = affine.apply #map29(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown85(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf32>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map28(%arg1)
      %4 = affine.apply #map29(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
    }
    return %0 : memref<128x64x3x3xf32>
  }
  func private @Unknown86(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map30(%arg1)
      %4 = affine.apply #map31(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown87(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf32>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map32(%arg1)
      %2 = affine.apply #map33(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func private @Unknown88(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map30(%arg1)
      %4 = affine.apply #map31(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown89(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map30(%arg1)
      %4 = affine.apply #map31(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown90(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf32>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map30(%arg1)
      %4 = affine.apply #map31(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
    }
    return %0 : memref<256x128x3x3xf32>
  }
  func private @Unknown91(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown92(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf32>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map36(%arg1)
      %2 = affine.apply #map37(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func private @Unknown93(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown94(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown95(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf32>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
    }
    return %0 : memref<512x256x3x3xf32>
  }
  func private @Unknown96(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map38(%arg1)
      %4 = affine.apply #map39(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown97(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf32>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map40(%arg1)
      %2 = affine.apply #map41(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func private @Unknown98(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map38(%arg1)
      %4 = affine.apply #map39(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown99(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.apply #map38(%arg1)
      %4 = affine.apply #map39(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<128xf32>, %arg11: memref<128xf32>, %arg12: memref<128xf32>, %arg13: memref<128xf32>, %arg14: memref<128xf32>, %arg15: memref<128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<256xf32>, %arg21: memref<256xf32>, %arg22: memref<256xf32>, %arg23: memref<256xf32>, %arg24: memref<256xf32>, %arg25: memref<256xf32>, %arg26: memref<256xf32>, %arg27: memref<256xf32>, %arg28: memref<256xf32>, %arg29: memref<256xf32>, %arg30: memref<512xf32>, %arg31: memref<512xf32>, %arg32: memref<512xf32>, %arg33: memref<512xf32>, %arg34: memref<512xf32>, %arg35: memref<512xf32>, %arg36: memref<512xf32>, %arg37: memref<512xf32>, %arg38: memref<512xf32>, %arg39: memref<512xf32>, %arg40: memref<64xf32>, %arg41: memref<64xf32>, %arg42: memref<64xf32>, %arg43: memref<64xf32>, %arg44: memref<64xf32>, %arg45: memref<64xf32>, %arg46: memref<64xf32>, %arg47: memref<64xf32>, %arg48: memref<64xf32>, %arg49: memref<64xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<128xf32>, %arg53: memref<128xf32>, %arg54: memref<128xf32>, %arg55: memref<128xf32>, %arg56: memref<128xf32>, %arg57: memref<128xf32>, %arg58: memref<128xf32>, %arg59: memref<128xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<512xf32>, %arg71: memref<512xf32>, %arg72: memref<512xf32>, %arg73: memref<512xf32>, %arg74: memref<512xf32>, %arg75: memref<512xf32>, %arg76: memref<512xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<64x3x7x7xf16>, %arg81: memref<1x3x224x224xf16>, %arg82: memref<1x64x112x112xf16>, %arg83: memref<1x64x112x112xf16>, %arg84: memref<1x64x56x56xf16>, %arg85: memref<64x64x3x3xf16>, %arg86: memref<1x64x56x56xf16>, %arg87: memref<1x64x56x56xf16>, %arg88: memref<64x64x3x3xf16>, %arg89: memref<1x64x56x56xf16>, %arg90: memref<1x64x56x56xf16>, %arg91: memref<64x64x3x3xf16>, %arg92: memref<1x64x56x56xf16>, %arg93: memref<1x64x56x56xf16>, %arg94: memref<64x64x3x3xf16>, %arg95: memref<1x64x56x56xf16>, %arg96: memref<1x64x56x56xf16>, %arg97: memref<128x64x3x3xf16>, %arg98: memref<1x128x28x28xf16>, %arg99: memref<1x128x28x28xf16>, %arg100: memref<128x128x3x3xf16>, %arg101: memref<1x128x28x28xf16>, %arg102: memref<128x64x1x1xf16>, %arg103: memref<1x128x28x28xf16>, %arg104: memref<1x128x28x28xf16>, %arg105: memref<128x128x3x3xf16>, %arg106: memref<1x128x28x28xf16>, %arg107: memref<1x128x28x28xf16>, %arg108: memref<128x128x3x3xf16>, %arg109: memref<1x128x28x28xf16>, %arg110: memref<1x128x28x28xf16>, %arg111: memref<256x128x3x3xf16>, %arg112: memref<1x256x14x14xf16>, %arg113: memref<1x256x14x14xf16>, %arg114: memref<256x256x3x3xf16>, %arg115: memref<1x256x14x14xf16>, %arg116: memref<256x128x1x1xf16>, %arg117: memref<1x256x14x14xf16>, %arg118: memref<1x256x14x14xf16>, %arg119: memref<256x256x3x3xf16>, %arg120: memref<1x256x14x14xf16>, %arg121: memref<1x256x14x14xf16>, %arg122: memref<256x256x3x3xf16>, %arg123: memref<1x256x14x14xf16>, %arg124: memref<1x256x14x14xf16>, %arg125: memref<512x256x3x3xf16>, %arg126: memref<1x512x7x7xf16>, %arg127: memref<1x512x7x7xf16>, %arg128: memref<512x512x3x3xf16>, %arg129: memref<1x512x7x7xf16>, %arg130: memref<512x256x1x1xf16>, %arg131: memref<1x512x7x7xf16>, %arg132: memref<1x512x7x7xf16>, %arg133: memref<512x512x3x3xf16>, %arg134: memref<1x512x7x7xf16>, %arg135: memref<1x512x7x7xf16>, %arg136: memref<512x512x3x3xf16>, %arg137: memref<1x512x7x7xf16>, %arg138: memref<1x512x7x7xf16>, %arg139: memref<1x512xf16>, %arg140: memref<512x1000xf16>, %arg141: memref<1x1000xf16>) -> (memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>) {
    %0 = memref.alloc() : memref<f32>
    %1 = memref.alloc() : memref<1000x512xf16>
    %2 = memref.alloc() : memref<1000xf32>
    %3 = memref.alloc() : memref<1x64x112x112xf16>
    %4 = memref.alloc() : memref<1x512xf16>
    %5 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%5) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.dot"(%arg141, %arg140, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>) -> ()
    %6 = call @Unknown0(%4, %arg138) : (memref<1x512xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %7:3 = call @BatchNormGradOp1(%arg137, %arg39, %6) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %8 = call @ConvBackwardDataOp2(%7#0, %arg136) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %9 = call @ConvBackwardFilterOp3(%arg135, %7#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %10 = call @Unknown4(%arg135, %8) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %11:3 = call @BatchNormGradOp5(%arg134, %arg37, %10) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %12 = call @ConvBackwardDataOp6(%11#0, %arg133) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %13 = call @ConvBackwardFilterOp7(%arg132, %11#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %14 = call @Unknown8(%6, %12, %arg132) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %15:3 = call @BatchNormGradOp9(%arg129, %arg33, %14) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %16 = call @ConvBackwardDataOp10(%15#0, %arg128) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %17 = call @ConvBackwardFilterOp11(%arg127, %15#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %18 = call @Unknown12(%arg127, %16) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %19:3 = call @BatchNormGradOp13(%arg126, %arg31, %18) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %20 = call @ConvBackwardDataOp14(%19#0, %arg125) : (memref<1x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %21 = call @ConvBackwardFilterOp15(%arg124, %19#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %22:3 = call @BatchNormGradOp16(%arg131, %arg35, %14) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %23 = call @ConvBackwardDataOp17(%22#0, %arg130) : (memref<1x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16>
    %24 = call @ConvBackwardFilterOp18(%arg124, %22#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %25 = call @Unknown19(%23, %20, %arg124) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %26:3 = call @BatchNormGradOp20(%arg123, %arg29, %25) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %27 = call @ConvBackwardDataOp21(%26#0, %arg122) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %28 = call @ConvBackwardFilterOp22(%arg121, %26#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %29 = call @Unknown23(%arg121, %27) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %30:3 = call @BatchNormGradOp24(%arg120, %arg27, %29) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %31 = call @ConvBackwardDataOp25(%30#0, %arg119) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %32 = call @ConvBackwardFilterOp26(%arg118, %30#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %33 = call @Unknown27(%25, %31, %arg118) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %34:3 = call @BatchNormGradOp28(%arg115, %arg23, %33) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %35 = call @ConvBackwardDataOp29(%34#0, %arg114) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %36 = call @ConvBackwardFilterOp30(%arg113, %34#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %37 = call @Unknown31(%arg113, %35) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %38:3 = call @BatchNormGradOp32(%arg112, %arg21, %37) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %39 = call @ConvBackwardDataOp33(%38#0, %arg111) : (memref<1x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %40 = call @ConvBackwardFilterOp34(%arg110, %38#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %41:3 = call @BatchNormGradOp35(%arg117, %arg25, %33) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %42 = call @ConvBackwardDataOp36(%41#0, %arg116) : (memref<1x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16>
    %43 = call @ConvBackwardFilterOp37(%arg110, %41#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %44 = call @Unknown38(%42, %39, %arg110) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %45:3 = call @BatchNormGradOp39(%arg109, %arg19, %44) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %46 = call @ConvBackwardDataOp40(%45#0, %arg108) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %47 = call @ConvBackwardFilterOp41(%arg107, %45#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %48 = call @Unknown42(%arg107, %46) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %49:3 = call @BatchNormGradOp43(%arg106, %arg17, %48) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %50 = call @ConvBackwardDataOp44(%49#0, %arg105) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %51 = call @ConvBackwardFilterOp45(%arg104, %49#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %52 = call @Unknown46(%44, %50, %arg104) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %53:3 = call @BatchNormGradOp47(%arg101, %arg13, %52) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %54 = call @ConvBackwardDataOp48(%53#0, %arg100) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %55 = call @ConvBackwardFilterOp49(%arg99, %53#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %56 = call @Unknown50(%arg99, %54) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %57:3 = call @BatchNormGradOp51(%arg98, %arg11, %56) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %58 = call @ConvBackwardDataOp52(%57#0, %arg97) : (memref<1x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %59 = call @ConvBackwardFilterOp53(%arg96, %57#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %60:3 = call @BatchNormGradOp54(%arg103, %arg15, %52) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %61 = call @ConvBackwardDataOp55(%60#0, %arg102) : (memref<1x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16>
    %62 = call @ConvBackwardFilterOp56(%arg96, %60#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %63 = call @Unknown57(%61, %58, %arg96) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %64:3 = call @BatchNormGradOp58(%arg95, %arg9, %63) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %65 = call @ConvBackwardDataOp59(%64#0, %arg94) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %66 = call @ConvBackwardFilterOp60(%arg93, %64#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %67 = call @Unknown61(%arg93, %65) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %68:3 = call @BatchNormGradOp62(%arg92, %arg7, %67) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %69 = call @ConvBackwardDataOp63(%68#0, %arg91) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %70 = call @ConvBackwardFilterOp64(%arg90, %68#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %71 = call @Unknown65(%63, %69, %arg90) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %72:3 = call @BatchNormGradOp66(%arg89, %arg5, %71) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %73 = call @ConvBackwardDataOp67(%72#0, %arg88) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %74 = call @ConvBackwardFilterOp68(%arg87, %72#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %75 = call @Unknown69(%arg87, %73) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %76:3 = call @BatchNormGradOp70(%arg86, %arg3, %75) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %77 = call @ConvBackwardDataOp71(%76#0, %arg85) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %78 = call @ConvBackwardFilterOp72(%arg84, %76#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %79 = call @Unknown73(%71, %77) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    "lmhlo.select_and_scatter"(%arg83, %79, %5, %3) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %106 = "mhlo.compare"(%arg142, %arg143) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%106) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %106 = mhlo.add %arg142, %arg143 : tensor<f16>
      "mhlo.return"(%106) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<f16>, memref<1x64x112x112xf16>) -> ()
    %80 = call @Unknown74(%arg83, %3) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %81:3 = call @BatchNormGradOp75(%arg82, %arg1, %80) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %82 = call @ConvBackwardFilterOp76(%arg81, %81#0) : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %83 = call @Unknown77(%82) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %84 = call @Unknown78(%arg141) : (memref<1x1000xf16>) -> memref<1x1000xf32>
    "lmhlo.reduce"(%84, %0, %2) ({
    ^bb0(%arg142: memref<f32>, %arg143: memref<f32>, %arg144: memref<f32>):
      "lmhlo.add"(%arg142, %arg143, %arg144) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<1x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %85 = call @Unknown79(%2) : (memref<1000xf32>) -> memref<1000xf32>
    "lmhlo.dot"(%arg141, %arg139, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x1000xf16>, memref<1x512xf16>, memref<1000x512xf16>) -> ()
    %86 = call @Unknown80(%1) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %87 = call @Unknown81(%78) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %88 = call @Unknown82(%74) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %89 = call @Unknown83(%70) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %90 = call @Unknown84(%66) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %91 = call @Unknown85(%59) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %92 = call @Unknown86(%55) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %93 = call @Unknown87(%62) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %94 = call @Unknown88(%51) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %95 = call @Unknown89(%47) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %96 = call @Unknown90(%40) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %97 = call @Unknown91(%36) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %98 = call @Unknown92(%43) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %99 = call @Unknown93(%32) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %100 = call @Unknown94(%28) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %101 = call @Unknown95(%21) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %102 = call @Unknown96(%17) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %103 = call @Unknown97(%24) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %104 = call @Unknown98(%13) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %105 = call @Unknown99(%9) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    return %81#2, %81#1, %83, %85, %86, %76#2, %76#1, %72#2, %72#1, %87, %88, %68#2, %68#1, %64#2, %64#1, %89, %90, %57#2, %57#1, %53#2, %53#1, %91, %92, %93, %60#2, %60#1, %49#2, %49#1, %45#2, %45#1, %94, %95, %38#2, %38#1, %34#2, %34#1, %96, %97, %98, %41#2, %41#1, %30#2, %30#1, %26#2, %26#1, %99, %100, %19#2, %19#1, %15#2, %15#1, %101, %102, %103, %22#2, %22#1, %11#2, %11#1, %7#2, %7#1, %104, %105 : memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>
  }
}

