// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0) -> (d0 mod 224)>
#map1 = affine_map<(d0) -> ((d0 floordiv 224) mod 224)>
#map2 = affine_map<(d0) -> (((d0 floordiv 224) floordiv 224) mod 3)>
#map3 = affine_map<(d0) -> (((d0 floordiv 224) floordiv 224) floordiv 3)>
#map4 = affine_map<(d0) -> (d0 mod 7)>
#map5 = affine_map<(d0) -> ((d0 floordiv 7) mod 7)>
#map6 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 3)>
#map7 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 3)>
#map8 = affine_map<(d0) -> (d0 mod 112)>
#map9 = affine_map<(d0) -> ((d0 floordiv 112) mod 112)>
#map10 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) mod 64)>
#map11 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) floordiv 64)>
#map12 = affine_map<(d0) -> (d0 mod 3)>
#map13 = affine_map<(d0) -> ((d0 floordiv 3) mod 3)>
#map14 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 64)>
#map15 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 64)>
#map16 = affine_map<(d0) -> (d0 mod 56)>
#map17 = affine_map<(d0) -> ((d0 floordiv 56) mod 56)>
#map18 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) mod 64)>
#map19 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) floordiv 64)>
#map20 = affine_map<(d0) -> (d0 mod 64)>
#map21 = affine_map<(d0) -> (d0 floordiv 64)>
#map22 = affine_map<(d0) -> (d0 mod 28)>
#map23 = affine_map<(d0) -> ((d0 floordiv 28) mod 28)>
#map24 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) mod 128)>
#map25 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) floordiv 128)>
#map26 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 128)>
#map27 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 128)>
#map28 = affine_map<(d0) -> (d0 mod 128)>
#map29 = affine_map<(d0) -> (d0 floordiv 128)>
#map30 = affine_map<(d0) -> (d0 mod 14)>
#map31 = affine_map<(d0) -> ((d0 floordiv 14) mod 14)>
#map32 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) mod 256)>
#map33 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) floordiv 256)>
#map34 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 256)>
#map35 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 256)>
#map36 = affine_map<(d0) -> (d0 mod 256)>
#map37 = affine_map<(d0) -> (d0 floordiv 256)>
#map38 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 512)>
#map39 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 512)>
#map40 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 512)>
#map41 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 512)>
#map42 = affine_map<(d0) -> (d0 mod 512)>
#map43 = affine_map<(d0) -> (d0 floordiv 512)>
#map44 = affine_map<(d0) -> (d0 mod 1000)>
#map45 = affine_map<(d0) -> (d0 floordiv 1000)>
module {
  func private @Unknown0(%arg0: memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1x3x224x224xf16>
    affine.for %arg1 = 0 to 150528 {
      %1 = affine.apply #map0(%arg1)
      %2 = affine.apply #map1(%arg1)
      %3 = affine.apply #map2(%arg1)
      %4 = affine.apply #map3(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x3x224x224xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x3x224x224xf16>
    }
    return %0 : memref<1x3x224x224xf16>
  }
  func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x3x7x7xf16>
    affine.for %arg1 = 0 to 9408 {
      %1 = affine.apply #map4(%arg1)
      %2 = affine.apply #map5(%arg1)
      %3 = affine.apply #map6(%arg1)
      %4 = affine.apply #map7(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x3x7x7xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x3x7x7xf16>
    }
    return %0 : memref<64x3x7x7xf16>
  }
  func private @BatchNormTrainingOp2(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x112x112xf32>
    %1 = memref.alloc() : memref<1x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown3(%arg0: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x112x112xf16>
    affine.for %arg1 = 0 to 802816 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
    }
    return %0 : memref<1x64x112x112xf16>
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @BatchNormTrainingOp5(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown6(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg1 = 0 to 200704 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown7(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @BatchNormTrainingOp8(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map16(%arg2)
      %2 = affine.apply #map17(%arg2)
      %3 = affine.apply #map18(%arg2)
      %4 = affine.apply #map19(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown10(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @BatchNormTrainingOp11(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown12(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg1 = 0 to 200704 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown13(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @BatchNormTrainingOp14(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown15(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map16(%arg2)
      %2 = affine.apply #map17(%arg2)
      %3 = affine.apply #map18(%arg2)
      %4 = affine.apply #map19(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown16(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<128x64x1x1xf16>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map20(%arg1)
      %2 = affine.apply #map21(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
    }
    return %0 : memref<128x64x1x1xf16>
  }
  func private @BatchNormTrainingOp17(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown18(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf16>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
    }
    return %0 : memref<128x64x3x3xf16>
  }
  func private @BatchNormTrainingOp19(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown20(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg1 = 0 to 100352 {
      %1 = affine.apply #map22(%arg1)
      %2 = affine.apply #map23(%arg1)
      %3 = affine.apply #map24(%arg1)
      %4 = affine.apply #map25(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown21(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map26(%arg1)
      %4 = affine.apply #map27(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @BatchNormTrainingOp22(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map22(%arg2)
      %2 = affine.apply #map23(%arg2)
      %3 = affine.apply #map24(%arg2)
      %4 = affine.apply #map25(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown24(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map26(%arg1)
      %4 = affine.apply #map27(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @BatchNormTrainingOp25(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown26(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg1 = 0 to 100352 {
      %1 = affine.apply #map22(%arg1)
      %2 = affine.apply #map23(%arg1)
      %3 = affine.apply #map24(%arg1)
      %4 = affine.apply #map25(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown27(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map26(%arg1)
      %4 = affine.apply #map27(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @BatchNormTrainingOp28(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown29(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map22(%arg2)
      %2 = affine.apply #map23(%arg2)
      %3 = affine.apply #map24(%arg2)
      %4 = affine.apply #map25(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown30(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<256x128x1x1xf16>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map28(%arg1)
      %2 = affine.apply #map29(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
    }
    return %0 : memref<256x128x1x1xf16>
  }
  func private @BatchNormTrainingOp31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown32(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf16>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map26(%arg1)
      %4 = affine.apply #map27(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
    }
    return %0 : memref<256x128x3x3xf16>
  }
  func private @BatchNormTrainingOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown34(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg1 = 0 to 50176 {
      %1 = affine.apply #map30(%arg1)
      %2 = affine.apply #map31(%arg1)
      %3 = affine.apply #map32(%arg1)
      %4 = affine.apply #map33(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown35(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @BatchNormTrainingOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map30(%arg2)
      %2 = affine.apply #map31(%arg2)
      %3 = affine.apply #map32(%arg2)
      %4 = affine.apply #map33(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown38(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @BatchNormTrainingOp39(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown40(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg1 = 0 to 50176 {
      %1 = affine.apply #map30(%arg1)
      %2 = affine.apply #map31(%arg1)
      %3 = affine.apply #map32(%arg1)
      %4 = affine.apply #map33(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown41(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @BatchNormTrainingOp42(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown43(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map30(%arg2)
      %2 = affine.apply #map31(%arg2)
      %3 = affine.apply #map32(%arg2)
      %4 = affine.apply #map33(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown44(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<512x256x1x1xf16>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map36(%arg1)
      %2 = affine.apply #map37(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
    }
    return %0 : memref<512x256x1x1xf16>
  }
  func private @BatchNormTrainingOp45(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown46(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf16>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map34(%arg1)
      %4 = affine.apply #map35(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
    }
    return %0 : memref<512x256x3x3xf16>
  }
  func private @BatchNormTrainingOp47(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown48(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg1 = 0 to 25088 {
      %1 = affine.apply #map4(%arg1)
      %2 = affine.apply #map5(%arg1)
      %3 = affine.apply #map38(%arg1)
      %4 = affine.apply #map39(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown49(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map40(%arg1)
      %4 = affine.apply #map41(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @BatchNormTrainingOp50(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map38(%arg2)
      %4 = affine.apply #map39(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown52(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map40(%arg1)
      %4 = affine.apply #map41(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @BatchNormTrainingOp53(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown54(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg1 = 0 to 25088 {
      %1 = affine.apply #map4(%arg1)
      %2 = affine.apply #map5(%arg1)
      %3 = affine.apply #map38(%arg1)
      %4 = affine.apply #map39(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = arith.maxf %5, %cst : f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown55(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.apply #map40(%arg1)
      %4 = affine.apply #map41(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @BatchNormTrainingOp56(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown57(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map38(%arg2)
      %4 = affine.apply #map39(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.addf %5, %6 : f16
      %8 = arith.maxf %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown58(%arg0: memref<1x512xf16>) -> memref<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = memref.alloc() : memref<1x512xf16>
    affine.for %arg1 = 0 to 512 {
      %1 = affine.apply #map42(%arg1)
      %2 = affine.apply #map43(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1x512xf16>
      %4 = arith.mulf %3, %cst : f16
      affine.store %4, %0[%2, %1] : memref<1x512xf16>
    }
    return %0 : memref<1x512xf16>
  }
  func private @Unknown59(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf16>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map42(%arg1)
      %2 = affine.apply #map43(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1] : memref<1000x512xf16>
    }
    return %0 : memref<1000x512xf16>
  }
  func private @Unknown60(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf16>
    %1 = memref.alloc() : memref<1x1000xf16>
    affine.for %arg2 = 0 to 1000 {
      %2 = affine.load %arg0[%arg2] : memref<1000xf32>
      %3 = arith.truncf %2 : f32 to f16
      affine.store %3, %0[%arg2] : memref<1000xf16>
      %4 = affine.apply #map44(%arg2)
      %5 = affine.apply #map45(%arg2)
      %6 = affine.load %arg1[%5, %4] : memref<1x1000xf16>
      %7 = affine.load %0[%4] : memref<1000xf16>
      %8 = arith.addf %6, %7 : f16
      affine.store %8, %1[%5, %4] : memref<1x1000xf16>
    }
    return %1 : memref<1x1000xf16>
  }
  func private @Unknown61(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown63(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown64(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown65(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown66(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown67(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown68(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown69(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown70(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 64 {
      %1 = affine.load %arg0[%arg2] : memref<64xf32>
      %2 = affine.load %arg1[%arg2] : memref<64xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @Unknown71(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown73(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown74(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown75(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown76(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown77(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown78(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown79(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown80(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 128 {
      %1 = affine.load %arg0[%arg2] : memref<128xf32>
      %2 = affine.load %arg1[%arg2] : memref<128xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @Unknown81(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown83(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown84(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown85(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown86(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown87(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown88(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown89(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown90(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 256 {
      %1 = affine.load %arg0[%arg2] : memref<256xf32>
      %2 = affine.load %arg1[%arg2] : memref<256xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown91(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown93(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown94(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown95(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown96(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown97(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown98(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown99(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @Unknown100(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.899999976 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e-01 : f32
    affine.for %arg2 = 0 to 512 {
      %1 = affine.load %arg0[%arg2] : memref<512xf32>
      %2 = affine.load %arg1[%arg2] : memref<512xf32>
      %3 = arith.mulf %2, %cst : f32
      %4 = arith.mulf %1, %cst_0 : f32
      %5 = arith.addf %4, %3 : f32
      affine.store %5, %0[%arg2] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<1000xf32>, %arg4: memref<1000x512xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64x64x3x3xf32>, %arg10: memref<64x64x3x3xf32>, %arg11: memref<64xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64x64x3x3xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<128xf32>, %arg21: memref<128x64x3x3xf32>, %arg22: memref<128x128x3x3xf32>, %arg23: memref<128x64x1x1xf32>, %arg24: memref<128xf32>, %arg25: memref<128xf32>, %arg26: memref<128xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128x128x3x3xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<256xf32>, %arg33: memref<256xf32>, %arg34: memref<256xf32>, %arg35: memref<256xf32>, %arg36: memref<256x128x3x3xf32>, %arg37: memref<256x256x3x3xf32>, %arg38: memref<256x128x1x1xf32>, %arg39: memref<256xf32>, %arg40: memref<256xf32>, %arg41: memref<256xf32>, %arg42: memref<256xf32>, %arg43: memref<256xf32>, %arg44: memref<256xf32>, %arg45: memref<256x256x3x3xf32>, %arg46: memref<256x256x3x3xf32>, %arg47: memref<512xf32>, %arg48: memref<512xf32>, %arg49: memref<512xf32>, %arg50: memref<512xf32>, %arg51: memref<512x256x3x3xf32>, %arg52: memref<512x512x3x3xf32>, %arg53: memref<512x256x1x1xf32>, %arg54: memref<512xf32>, %arg55: memref<512xf32>, %arg56: memref<512xf32>, %arg57: memref<512xf32>, %arg58: memref<512xf32>, %arg59: memref<512xf32>, %arg60: memref<512x512x3x3xf32>, %arg61: memref<512x512x3x3xf32>, %arg62: memref<i64>, %arg63: memref<64xf32>, %arg64: memref<64xf32>, %arg65: memref<i64>, %arg66: memref<64xf32>, %arg67: memref<64xf32>, %arg68: memref<i64>, %arg69: memref<64xf32>, %arg70: memref<64xf32>, %arg71: memref<i64>, %arg72: memref<64xf32>, %arg73: memref<64xf32>, %arg74: memref<i64>, %arg75: memref<64xf32>, %arg76: memref<64xf32>, %arg77: memref<i64>, %arg78: memref<128xf32>, %arg79: memref<128xf32>, %arg80: memref<i64>, %arg81: memref<128xf32>, %arg82: memref<128xf32>, %arg83: memref<i64>, %arg84: memref<128xf32>, %arg85: memref<128xf32>, %arg86: memref<i64>, %arg87: memref<128xf32>, %arg88: memref<128xf32>, %arg89: memref<i64>, %arg90: memref<128xf32>, %arg91: memref<128xf32>, %arg92: memref<i64>, %arg93: memref<256xf32>, %arg94: memref<256xf32>, %arg95: memref<i64>, %arg96: memref<256xf32>, %arg97: memref<256xf32>, %arg98: memref<i64>, %arg99: memref<256xf32>, %arg100: memref<256xf32>, %arg101: memref<i64>, %arg102: memref<256xf32>, %arg103: memref<256xf32>, %arg104: memref<i64>, %arg105: memref<256xf32>, %arg106: memref<256xf32>, %arg107: memref<i64>, %arg108: memref<512xf32>, %arg109: memref<512xf32>, %arg110: memref<i64>, %arg111: memref<512xf32>, %arg112: memref<512xf32>, %arg113: memref<i64>, %arg114: memref<512xf32>, %arg115: memref<512xf32>, %arg116: memref<i64>, %arg117: memref<512xf32>, %arg118: memref<512xf32>, %arg119: memref<i64>, %arg120: memref<512xf32>, %arg121: memref<512xf32>, %arg122: memref<1x3x224x224xf32>) -> (memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>) {
    %0 = memref.alloc() : memref<f16>
    %1 = memref.alloc() : memref<1x1000xf16>
    %2 = memref.alloc() : memref<512x1000xf16>
    %3 = memref.alloc() : memref<1x512xf16>
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    %5 = memref.alloc() : memref<1x512x7x7xf16>
    %6 = memref.alloc() : memref<1x512x7x7xf16>
    %7 = memref.alloc() : memref<1x512x7x7xf16>
    %8 = memref.alloc() : memref<1x512x7x7xf16>
    %9 = memref.alloc() : memref<1x256x14x14xf16>
    %10 = memref.alloc() : memref<1x256x14x14xf16>
    %11 = memref.alloc() : memref<1x256x14x14xf16>
    %12 = memref.alloc() : memref<1x256x14x14xf16>
    %13 = memref.alloc() : memref<1x256x14x14xf16>
    %14 = memref.alloc() : memref<1x128x28x28xf16>
    %15 = memref.alloc() : memref<1x128x28x28xf16>
    %16 = memref.alloc() : memref<1x128x28x28xf16>
    %17 = memref.alloc() : memref<1x128x28x28xf16>
    %18 = memref.alloc() : memref<1x128x28x28xf16>
    %19 = memref.alloc() : memref<1x64x56x56xf16>
    %20 = memref.alloc() : memref<1x64x56x56xf16>
    %21 = memref.alloc() : memref<1x64x56x56xf16>
    %22 = memref.alloc() : memref<1x64x56x56xf16>
    %23 = memref.alloc() : memref<1x64x56x56xf16>
    %24 = memref.alloc() : memref<1x64x112x112xf16>
    %25 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.constant"(%25) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %26 = call @Unknown0(%arg122) : (memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16>
    %27 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    lmhlo.convolution(%26, %27, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x3x224x224xf16>, memref<64x3x7x7xf16>, memref<1x64x112x112xf16>) -> ()
    %28:3 = call @BatchNormTrainingOp2(%24, %arg1, %arg0) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %29 = call @Unknown3(%28#0) : (memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    "lmhlo.reduce_window"(%29, %25, %23) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):  // no predecessors
      %127 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg123, %arg124, %127) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%127, %arg125) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<f16>, memref<1x64x56x56xf16>) -> ()
    %30 = call @Unknown4(%arg9) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%23, %30, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %31:3 = call @BatchNormTrainingOp5(%22, %arg6, %arg5) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %32 = call @Unknown6(%31#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %33 = call @Unknown7(%arg10) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%32, %33, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %34:3 = call @BatchNormTrainingOp8(%21, %arg8, %arg7) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %35 = call @Unknown9(%34#0, %23) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %36 = call @Unknown10(%arg15) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%35, %36, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %37:3 = call @BatchNormTrainingOp11(%20, %arg12, %arg11) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %38 = call @Unknown12(%37#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %39 = call @Unknown13(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    lmhlo.convolution(%38, %39, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %40:3 = call @BatchNormTrainingOp14(%19, %arg14, %arg13) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %41 = call @Unknown15(%40#0, %35) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %42 = call @Unknown16(%arg23) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    lmhlo.convolution(%41, %42, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>) -> ()
    %43:3 = call @BatchNormTrainingOp17(%18, %arg25, %arg24) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %44 = call @Unknown18(%arg21) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    lmhlo.convolution(%41, %44, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %45:3 = call @BatchNormTrainingOp19(%17, %arg18, %arg17) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %46 = call @Unknown20(%45#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %47 = call @Unknown21(%arg22) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%46, %47, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %48:3 = call @BatchNormTrainingOp22(%16, %arg20, %arg19) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %49 = call @Unknown23(%48#0, %43#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %50 = call @Unknown24(%arg30) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%49, %50, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %51:3 = call @BatchNormTrainingOp25(%15, %arg27, %arg26) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %52 = call @Unknown26(%51#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %53 = call @Unknown27(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    lmhlo.convolution(%52, %53, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %54:3 = call @BatchNormTrainingOp28(%14, %arg29, %arg28) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %55 = call @Unknown29(%54#0, %49) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %56 = call @Unknown30(%arg38) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    lmhlo.convolution(%55, %56, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>) -> ()
    %57:3 = call @BatchNormTrainingOp31(%13, %arg40, %arg39) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %58 = call @Unknown32(%arg36) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    lmhlo.convolution(%55, %58, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %59:3 = call @BatchNormTrainingOp33(%12, %arg33, %arg32) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %60 = call @Unknown34(%59#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %61 = call @Unknown35(%arg37) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%60, %61, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %62:3 = call @BatchNormTrainingOp36(%11, %arg35, %arg34) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %63 = call @Unknown37(%62#0, %57#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %64 = call @Unknown38(%arg45) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%63, %64, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %65:3 = call @BatchNormTrainingOp39(%10, %arg42, %arg41) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %66 = call @Unknown40(%65#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %67 = call @Unknown41(%arg46) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    lmhlo.convolution(%66, %67, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %68:3 = call @BatchNormTrainingOp42(%9, %arg44, %arg43) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %69 = call @Unknown43(%68#0, %63) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %70 = call @Unknown44(%arg53) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    lmhlo.convolution(%69, %70, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>) -> ()
    %71:3 = call @BatchNormTrainingOp45(%8, %arg55, %arg54) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %72 = call @Unknown46(%arg51) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    lmhlo.convolution(%69, %72, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %73:3 = call @BatchNormTrainingOp47(%7, %arg48, %arg47) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %74 = call @Unknown48(%73#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %75 = call @Unknown49(%arg52) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%74, %75, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %76:3 = call @BatchNormTrainingOp50(%6, %arg50, %arg49) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %77 = call @Unknown51(%76#0, %71#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %78 = call @Unknown52(%arg60) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%77, %78, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %79:3 = call @BatchNormTrainingOp53(%5, %arg57, %arg56) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %80 = call @Unknown54(%79#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %81 = call @Unknown55(%arg61) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    lmhlo.convolution(%80, %81, %4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %82:3 = call @BatchNormTrainingOp56(%4, %arg59, %arg58) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %83 = call @Unknown57(%82#0, %77) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    "lmhlo.reduce"(%83, %0, %3) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg123, %arg124, %arg125) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<1x512x7x7xf16>, memref<f16>, memref<1x512xf16>) -> ()
    %84 = call @Unknown58(%3) : (memref<1x512xf16>) -> memref<1x512xf16>
    %85 = call @Unknown59(%arg4) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    "lmhlo.transpose"(%85, %2) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<1000x512xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.dot"(%84, %85, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512xf16>, memref<1000x512xf16>, memref<1x1000xf16>) -> ()
    %86 = call @Unknown60(%arg3, %1) : (memref<1000xf32>, memref<1x1000xf16>) -> memref<1x1000xf16>
    %87 = call @Unknown61(%28#1, %arg63) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %88 = call @Unknown62(%28#2, %arg64) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %89 = call @Unknown63(%31#1, %arg66) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %90 = call @Unknown64(%31#2, %arg67) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %91 = call @Unknown65(%34#1, %arg69) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %92 = call @Unknown66(%34#2, %arg70) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %93 = call @Unknown67(%37#1, %arg72) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %94 = call @Unknown68(%37#2, %arg73) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %95 = call @Unknown69(%40#1, %arg75) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %96 = call @Unknown70(%40#2, %arg76) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %97 = call @Unknown71(%45#1, %arg78) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %98 = call @Unknown72(%45#2, %arg79) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %99 = call @Unknown73(%48#1, %arg81) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %100 = call @Unknown74(%48#2, %arg82) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %101 = call @Unknown75(%43#1, %arg84) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %102 = call @Unknown76(%43#2, %arg85) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %103 = call @Unknown77(%51#1, %arg87) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %104 = call @Unknown78(%51#2, %arg88) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %105 = call @Unknown79(%54#1, %arg90) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %106 = call @Unknown80(%54#2, %arg91) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %107 = call @Unknown81(%59#1, %arg93) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %108 = call @Unknown82(%59#2, %arg94) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %109 = call @Unknown83(%62#1, %arg96) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %110 = call @Unknown84(%62#2, %arg97) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %111 = call @Unknown85(%57#1, %arg99) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %112 = call @Unknown86(%57#2, %arg100) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %113 = call @Unknown87(%65#1, %arg102) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %114 = call @Unknown88(%65#2, %arg103) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %115 = call @Unknown89(%68#1, %arg105) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %116 = call @Unknown90(%68#2, %arg106) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %117 = call @Unknown91(%73#1, %arg108) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %118 = call @Unknown92(%73#2, %arg109) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %119 = call @Unknown93(%76#1, %arg111) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %120 = call @Unknown94(%76#2, %arg112) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %121 = call @Unknown95(%71#1, %arg114) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %122 = call @Unknown96(%71#2, %arg115) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %123 = call @Unknown97(%79#1, %arg117) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %124 = call @Unknown98(%79#2, %arg118) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %125 = call @Unknown99(%82#1, %arg120) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %126 = call @Unknown100(%82#2, %arg121) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    return %86, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %27, %26, %24, %29, %23, %30, %22, %32, %33, %21, %35, %36, %20, %38, %39, %19, %41, %44, %17, %46, %47, %16, %42, %18, %49, %50, %15, %52, %53, %14, %55, %58, %12, %60, %61, %11, %56, %13, %63, %64, %10, %66, %67, %9, %69, %72, %7, %74, %75, %6, %70, %8, %77, %78, %5, %80, %81, %4, %83, %84, %2 : memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>
  }
}

