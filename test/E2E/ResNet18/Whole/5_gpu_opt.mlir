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
#map8 = affine_map<(d0) -> (d0 mod 3)>
#map9 = affine_map<(d0) -> ((d0 floordiv 3) mod 3)>
#map10 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 64)>
#map11 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 64)>
#map12 = affine_map<(d0) -> (d0 mod 64)>
#map13 = affine_map<(d0) -> (d0 floordiv 64)>
#map14 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 128)>
#map15 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 128)>
#map16 = affine_map<(d0) -> (d0 mod 128)>
#map17 = affine_map<(d0) -> (d0 floordiv 128)>
#map18 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 256)>
#map19 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 256)>
#map20 = affine_map<(d0) -> (d0 mod 256)>
#map21 = affine_map<(d0) -> (d0 floordiv 256)>
#map22 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 512)>
#map23 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 512)>
#map24 = affine_map<(d0) -> (d0 mod 1000)>
#map25 = affine_map<(d0) -> (d0 floordiv 1000)>
#map26 = affine_map<(d0) -> (d0 mod 512)>
#map27 = affine_map<(d0) -> (d0 floordiv 512)>
#map28 = affine_map<(d0) -> (d0 mod 112)>
#map29 = affine_map<(d0) -> ((d0 floordiv 112) mod 112)>
#map30 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) mod 64)>
#map31 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) floordiv 64)>
#map32 = affine_map<(d0) -> (d0 mod 56)>
#map33 = affine_map<(d0) -> ((d0 floordiv 56) mod 56)>
#map34 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) mod 64)>
#map35 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) floordiv 64)>
#map36 = affine_map<(d0) -> (d0 mod 28)>
#map37 = affine_map<(d0) -> ((d0 floordiv 28) mod 28)>
#map38 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) mod 128)>
#map39 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) floordiv 128)>
#map40 = affine_map<(d0) -> (d0 mod 14)>
#map41 = affine_map<(d0) -> ((d0 floordiv 14) mod 14)>
#map42 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) mod 256)>
#map43 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) floordiv 256)>
#map44 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 512)>
#map45 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 512)>
module @IrToMhlo.2452 {
  func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x3x224x224xf16>
    affine.for %arg1 = 0 to 602112 {
      %1 = affine.apply #map0(%arg1)
      %2 = affine.apply #map1(%arg1)
      %3 = affine.apply #map2(%arg1)
      %4 = affine.apply #map3(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x3x224x224xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<4x3x224x224xf16>
    }
    return %0 : memref<4x3x224x224xf16>
  }
  func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf16>
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
  func private @BatchNormTrainingOp2(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x112x112xf32>
    %1 = memref.alloc() : memref<4x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %1 : memref<4x64x112x112xf16>
  }
  func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf16>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
    }
    return %0 : memref<128x64x1x1xf16>
  }
  func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf16>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
    }
    return %0 : memref<128x64x3x3xf16>
  }
  func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown10(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf16>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
    }
    return %0 : memref<256x128x1x1xf16>
  }
  func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf16>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
    }
    return %0 : memref<256x128x3x3xf16>
  }
  func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown15(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf16>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map20(%arg1)
      %2 = affine.apply #map21(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
    }
    return %0 : memref<512x256x1x1xf16>
  }
  func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf16>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
    }
    return %0 : memref<512x256x3x3xf16>
  }
  func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown20(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf16>
    affine.for %arg1 = 0 to 4000 {
      %1 = affine.apply #map24(%arg1)
      %2 = affine.apply #map25(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<4x1000xf32>
      %4 = arith.mulf %3, %cst : f32
      %5 = arith.truncf %4 : f32 to f16
      affine.store %5, %0[%2, %1] : memref<4x1000xf16>
    }
    return %0 : memref<4x1000xf16>
  }
  func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf16>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1] : memref<1000x512xf16>
    }
    return %0 : memref<1000x512xf16>
  }
  func private @Unknown24(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000xf16>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      affine.store %2, %0[%arg1] : memref<1000xf16>
    }
    return %0 : memref<1000xf16>
  }
  func private @Unknown25(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x112x112xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x64x112x112xf16>
    affine.for %arg1 = 0 to 3211264 {
      %2 = affine.apply #map28(%arg1)
      %3 = affine.apply #map29(%arg1)
      %4 = affine.apply #map30(%arg1)
      %5 = affine.apply #map31(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x64x112x112xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x64x112x112xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x64x112x112xi1>
    }
    return %1, %0 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func private @BatchNormTrainingOp26(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @Unknown27(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg1 = 0 to 802816 {
      %2 = affine.apply #map32(%arg1)
      %3 = affine.apply #map33(%arg1)
      %4 = affine.apply #map34(%arg1)
      %5 = affine.apply #map35(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x64x56x56xi1>
    }
    return %1, %0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @Unknown29(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg2 = 0 to 802816 {
      %2 = affine.apply #map32(%arg2)
      %3 = affine.apply #map33(%arg2)
      %4 = affine.apply #map34(%arg2)
      %5 = affine.apply #map35(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x64x56x56xi1>
    }
    return %1, %0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @Unknown31(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg1 = 0 to 802816 {
      %2 = affine.apply #map32(%arg1)
      %3 = affine.apply #map33(%arg1)
      %4 = affine.apply #map34(%arg1)
      %5 = affine.apply #map35(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x64x56x56xi1>
    }
    return %1, %0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp32(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @Unknown33(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg2 = 0 to 802816 {
      %2 = affine.apply #map32(%arg2)
      %3 = affine.apply #map33(%arg2)
      %4 = affine.apply #map34(%arg2)
      %5 = affine.apply #map35(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x64x56x56xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x64x56x56xi1>
    }
    return %1, %0 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp34(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @BatchNormTrainingOp35(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @Unknown36(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg1 = 0 to 401408 {
      %2 = affine.apply #map36(%arg1)
      %3 = affine.apply #map37(%arg1)
      %4 = affine.apply #map38(%arg1)
      %5 = affine.apply #map39(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x128x28x28xi1>
    }
    return %1, %0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @Unknown38(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg2 = 0 to 401408 {
      %2 = affine.apply #map36(%arg2)
      %3 = affine.apply #map37(%arg2)
      %4 = affine.apply #map38(%arg2)
      %5 = affine.apply #map39(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x128x28x28xi1>
    }
    return %1, %0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @Unknown40(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg1 = 0 to 401408 {
      %2 = affine.apply #map36(%arg1)
      %3 = affine.apply #map37(%arg1)
      %4 = affine.apply #map38(%arg1)
      %5 = affine.apply #map39(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x128x28x28xi1>
    }
    return %1, %0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp41(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @Unknown42(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg2 = 0 to 401408 {
      %2 = affine.apply #map36(%arg2)
      %3 = affine.apply #map37(%arg2)
      %4 = affine.apply #map38(%arg2)
      %5 = affine.apply #map39(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x128x28x28xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x128x28x28xi1>
    }
    return %1, %0 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp43(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @BatchNormTrainingOp44(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @Unknown45(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg1 = 0 to 200704 {
      %2 = affine.apply #map40(%arg1)
      %3 = affine.apply #map41(%arg1)
      %4 = affine.apply #map42(%arg1)
      %5 = affine.apply #map43(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x256x14x14xi1>
    }
    return %1, %0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @Unknown47(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg2 = 0 to 200704 {
      %2 = affine.apply #map40(%arg2)
      %3 = affine.apply #map41(%arg2)
      %4 = affine.apply #map42(%arg2)
      %5 = affine.apply #map43(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x256x14x14xi1>
    }
    return %1, %0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @Unknown49(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg1 = 0 to 200704 {
      %2 = affine.apply #map40(%arg1)
      %3 = affine.apply #map41(%arg1)
      %4 = affine.apply #map42(%arg1)
      %5 = affine.apply #map43(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x256x14x14xi1>
    }
    return %1, %0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp50(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @Unknown51(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg2 = 0 to 200704 {
      %2 = affine.apply #map40(%arg2)
      %3 = affine.apply #map41(%arg2)
      %4 = affine.apply #map42(%arg2)
      %5 = affine.apply #map43(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x256x14x14xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x256x14x14xi1>
    }
    return %1, %0 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp52(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @BatchNormTrainingOp53(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @Unknown54(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg1 = 0 to 100352 {
      %2 = affine.apply #map4(%arg1)
      %3 = affine.apply #map5(%arg1)
      %4 = affine.apply #map44(%arg1)
      %5 = affine.apply #map45(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x512x7x7xi1>
    }
    return %1, %0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @Unknown56(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg2 = 0 to 100352 {
      %2 = affine.apply #map4(%arg2)
      %3 = affine.apply #map5(%arg2)
      %4 = affine.apply #map44(%arg2)
      %5 = affine.apply #map45(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x512x7x7xi1>
    }
    return %1, %0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @Unknown58(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg1 = 0 to 100352 {
      %2 = affine.apply #map4(%arg1)
      %3 = affine.apply #map5(%arg1)
      %4 = affine.apply #map44(%arg1)
      %5 = affine.apply #map45(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<4x512x7x7xi1>
    }
    return %1, %0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp59(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @Unknown60(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg2 = 0 to 100352 {
      %2 = affine.apply #map4(%arg2)
      %3 = affine.apply #map5(%arg2)
      %4 = affine.apply #map44(%arg2)
      %5 = affine.apply #map45(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<4x512x7x7xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<4x512x7x7xi1>
    }
    return %1, %0 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @Unknown61(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512xf16>
    affine.for %arg1 = 0 to 2048 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<4x512xf16>
      %4 = arith.mulf %3, %cst : f16
      affine.store %4, %0[%2, %1] : memref<4x512xf16>
    }
    return %0 : memref<4x512xf16>
  }
  func private @Unknown62(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf16>
    affine.for %arg2 = 0 to 4000 {
      %1 = affine.apply #map24(%arg2)
      %2 = affine.apply #map25(%arg2)
      %3 = affine.load %arg1[%2, %1] : memref<4x1000xf16>
      %4 = affine.load %arg0[%1] : memref<1000xf16>
      %5 = arith.addf %3, %4 : f16
      affine.store %5, %0[%2, %1] : memref<4x1000xf16>
    }
    return %0 : memref<4x1000xf16>
  }
  func private @Unknown63(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf16>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf16>
    affine.for %arg2 = 0 to 4000 {
      %2 = affine.apply #map24(%arg2)
      %3 = affine.apply #map25(%arg2)
      %4 = affine.load %arg1[%3, %2] : memref<4x1000xf16>
      %5 = affine.load %arg0[%3] : memref<4xf16>
      %6 = arith.subf %4, %5 : f16
      affine.store %6, %0[%3, %2] : memref<4x1000xf16>
      %7 = arith.subf %4, %5 : f16
      %8 = math.exp %7 : f16
      affine.store %8, %1[%3, %2] : memref<4x1000xf16>
    }
    return %0, %1 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func private @Unknown64(%arg0: memref<4xf16>) -> memref<4xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4xf16>
    affine.for %arg1 = 0 to 4 {
      %1 = affine.load %arg0[%arg1] : memref<4xf16>
      %2 = math.log %1 : f16
      affine.store %2, %0[%arg1] : memref<4xf16>
    }
    return %0 : memref<4xf16>
  }
  func private @Unknown65(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf16>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<4x1000xf32>
    affine.for %arg5 = 0 to 4000 {
      %3 = affine.apply #map24(%arg5)
      %4 = affine.apply #map25(%arg5)
      %5 = affine.load %arg3[%4, %3] : memref<4x1000xf16>
      %6 = affine.load %arg1[%4, %3] : memref<4x1000xf16>
      %7 = affine.load %arg0[%4] : memref<4xf16>
      %8 = affine.load %arg2[%4] : memref<4xf16>
      %9 = arith.subf %6, %7 : f16
      %10 = math.exp %9 : f16
      %11 = arith.mulf %10, %8 : f16
      %12 = arith.subf %5, %11 : f16
      affine.store %12, %0[%4, %3] : memref<4x1000xf16>
      %13 = affine.load %arg4[%4, %3] : memref<4x1000xf32>
      %14 = arith.subf %6, %7 : f16
      %15 = arith.extf %14 : f16 to f32
      %16 = arith.mulf %15, %13 : f32
      affine.store %16, %2[%4, %3] : memref<4x1000xf32>
      %17 = arith.subf %6, %7 : f16
      %18 = math.exp %17 : f16
      %19 = arith.mulf %18, %8 : f16
      %20 = arith.subf %5, %19 : f16
      %21 = arith.extf %20 : f16 to f32
      affine.store %21, %1[%4, %3] : memref<4x1000xf32>
    }
    return %0, %2, %1 : memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>
  }
  func private @Unknown66(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    %cst_0 = arith.constant 4.900000e+01 : f16
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map44(%arg2)
      %4 = affine.apply #map45(%arg2)
      %5 = affine.load %arg1[%4, %3, %2, %1] : memref<4x512x7x7xi1>
      %6 = affine.load %arg0[%4, %3] : memref<4x512xf16>
      %7 = arith.divf %6, %cst_0 : f16
      %8 = arith.select %5, %7, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<4x512x7x7xf16>
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp67(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    %5 = memref.alloc() : memref<4x512x7x7xf32>
    %6 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp68(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp69(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown70(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map44(%arg2)
      %4 = affine.apply #map45(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x512x7x7xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x512x7x7xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x512x7x7xf16>
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp71(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    %5 = memref.alloc() : memref<4x512x7x7xf32>
    %6 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp72(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp73(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown74(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg3 = 0 to 100352 {
      %1 = affine.apply #map4(%arg3)
      %2 = affine.apply #map5(%arg3)
      %3 = affine.apply #map44(%arg3)
      %4 = affine.apply #map45(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x512x7x7xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x512x7x7xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x512x7x7xf16>
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp75(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    %5 = memref.alloc() : memref<4x512x7x7xf32>
    %6 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp76(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %1 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp77(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown78(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x512x7x7xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map44(%arg2)
      %4 = affine.apply #map45(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x512x7x7xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x512x7x7xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x512x7x7xf16>
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp79(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    %5 = memref.alloc() : memref<4x512x7x7xf32>
    %6 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @BatchNormGradOp82(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<4x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<4x512x7x7xf32>
    %5 = memref.alloc() : memref<4x512x7x7xf32>
    %6 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp83(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<1x1x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp84(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown85(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg3 = 0 to 200704 {
      %1 = affine.apply #map40(%arg3)
      %2 = affine.apply #map41(%arg3)
      %3 = affine.apply #map42(%arg3)
      %4 = affine.apply #map43(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x256x14x14xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp86(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    %5 = memref.alloc() : memref<4x256x14x14xf32>
    %6 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp87(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp88(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown89(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map40(%arg2)
      %2 = affine.apply #map41(%arg2)
      %3 = affine.apply #map42(%arg2)
      %4 = affine.apply #map43(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x256x14x14xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp90(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    %5 = memref.alloc() : memref<4x256x14x14xf32>
    %6 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp91(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp92(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown93(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg3 = 0 to 200704 {
      %1 = affine.apply #map40(%arg3)
      %2 = affine.apply #map41(%arg3)
      %3 = affine.apply #map42(%arg3)
      %4 = affine.apply #map43(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x256x14x14xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp94(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    %5 = memref.alloc() : memref<4x256x14x14xf32>
    %6 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown97(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x256x14x14xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map40(%arg2)
      %2 = affine.apply #map41(%arg2)
      %3 = affine.apply #map42(%arg2)
      %4 = affine.apply #map43(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x256x14x14xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x256x14x14xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x256x14x14xf16>
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp98(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    %5 = memref.alloc() : memref<4x256x14x14xf32>
    %6 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp99(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp100(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @BatchNormGradOp101(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<4x256x14x14xf32>
    %5 = memref.alloc() : memref<4x256x14x14xf32>
    %6 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp102(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<1x1x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp103(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown104(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg3 = 0 to 401408 {
      %1 = affine.apply #map36(%arg3)
      %2 = affine.apply #map37(%arg3)
      %3 = affine.apply #map38(%arg3)
      %4 = affine.apply #map39(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x128x28x28xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp105(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    %5 = memref.alloc() : memref<4x128x28x28xf32>
    %6 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp106(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp107(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown108(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg2 = 0 to 401408 {
      %1 = affine.apply #map36(%arg2)
      %2 = affine.apply #map37(%arg2)
      %3 = affine.apply #map38(%arg2)
      %4 = affine.apply #map39(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x128x28x28xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp109(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    %5 = memref.alloc() : memref<4x128x28x28xf32>
    %6 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown112(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg3 = 0 to 401408 {
      %1 = affine.apply #map36(%arg3)
      %2 = affine.apply #map37(%arg3)
      %3 = affine.apply #map38(%arg3)
      %4 = affine.apply #map39(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x128x28x28xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp113(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    %5 = memref.alloc() : memref<4x128x28x28xf32>
    %6 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp114(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp115(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown116(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x128x28x28xf16>
    affine.for %arg2 = 0 to 401408 {
      %1 = affine.apply #map36(%arg2)
      %2 = affine.apply #map37(%arg2)
      %3 = affine.apply #map38(%arg2)
      %4 = affine.apply #map39(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x128x28x28xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x128x28x28xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x128x28x28xf16>
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp117(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    %5 = memref.alloc() : memref<4x128x28x28xf32>
    %6 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp118(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp119(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @BatchNormGradOp120(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<4x128x28x28xf32>
    %5 = memref.alloc() : memref<4x128x28x28xf32>
    %6 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp121(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<1x1x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp122(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown123(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg3 = 0 to 802816 {
      %1 = affine.apply #map32(%arg3)
      %2 = affine.apply #map33(%arg3)
      %3 = affine.apply #map34(%arg3)
      %4 = affine.apply #map35(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x64x56x56xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp124(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    %5 = memref.alloc() : memref<4x64x56x56xf32>
    %6 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp125(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp126(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown127(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map32(%arg2)
      %2 = affine.apply #map33(%arg2)
      %3 = affine.apply #map34(%arg2)
      %4 = affine.apply #map35(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x56x56xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp128(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    %5 = memref.alloc() : memref<4x64x56x56xf32>
    %6 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp129(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp130(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown131(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg3 = 0 to 802816 {
      %1 = affine.apply #map32(%arg3)
      %2 = affine.apply #map33(%arg3)
      %3 = affine.apply #map34(%arg3)
      %4 = affine.apply #map35(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<4x64x56x56xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp132(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    %5 = memref.alloc() : memref<4x64x56x56xf32>
    %6 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp133(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp134(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown135(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map32(%arg2)
      %2 = affine.apply #map33(%arg2)
      %3 = affine.apply #map34(%arg2)
      %4 = affine.apply #map35(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x56x56xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp136(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x56x56xf32>
    %5 = memref.alloc() : memref<4x64x56x56xf32>
    %6 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp137(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp138(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown139(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x56x56xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map32(%arg2)
      %2 = affine.apply #map33(%arg2)
      %3 = affine.apply #map34(%arg2)
      %4 = affine.apply #map35(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x56x56xf16>
      %7 = arith.addf %5, %6 : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x64x56x56xf16>
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @Unknown140(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<4x64x112x112xf16>
    affine.for %arg2 = 0 to 3211264 {
      %1 = affine.apply #map28(%arg2)
      %2 = affine.apply #map29(%arg2)
      %3 = affine.apply #map30(%arg2)
      %4 = affine.apply #map31(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<4x64x112x112xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<4x64x112x112xf16>
      %7 = arith.select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<4x64x112x112xf16>
    }
    return %0 : memref<4x64x112x112xf16>
  }
  func private @BatchNormGradOp141(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<4x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<4x64x112x112xf32>
    %5 = memref.alloc() : memref<4x64x112x112xf32>
    %6 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp142(%arg0: memref<4x3x224x224xf16>, %arg1: memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown143(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
    affine.for %arg1 = 0 to 1 {
      %1 = affine.load %arg0[] : memref<f32>
      %2 = arith.negf %1 : f32
      %3 = arith.divf %2, %cst : f32
      affine.store %3, %0[] : memref<f32>
    }
    return %0 : memref<f32>
  }
  func private @Unknown144(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf32>
    affine.for %arg1 = 0 to 9408 {
      %1 = affine.apply #map4(%arg1)
      %2 = affine.apply #map5(%arg1)
      %3 = affine.apply #map6(%arg1)
      %4 = affine.apply #map7(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x3x7x7xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x3x7x7xf32>
    }
    return %0 : memref<64x3x7x7xf32>
  }
  func private @Unknown145(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown146(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown147(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown148(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown149(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf32>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map10(%arg1)
      %4 = affine.apply #map11(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
    }
    return %0 : memref<128x64x3x3xf32>
  }
  func private @Unknown150(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown151(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf32>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map12(%arg1)
      %2 = affine.apply #map13(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func private @Unknown152(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown153(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown154(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf32>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
    }
    return %0 : memref<256x128x3x3xf32>
  }
  func private @Unknown155(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown156(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf32>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func private @Unknown157(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown158(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown159(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf32>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
    }
    return %0 : memref<512x256x3x3xf32>
  }
  func private @Unknown160(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown161(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf32>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map20(%arg1)
      %2 = affine.apply #map21(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func private @Unknown162(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown163(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @MatmulOp164(%arg0: memref<4x512xf16>, %arg1: memref<4x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<512x1000xf16>
    %1 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512xf16>, memref<4x1000xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %1 : memref<1000x512xf16>
  }
  func private @Unknown165(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf32>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map26(%arg1)
      %2 = affine.apply #map27(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1000x512xf32>
    }
    return %0 : memref<1000x512xf32>
  }
  func private @Unknown166(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      affine.store %3, %0[%arg1] : memref<1000xf32>
    }
    return %0 : memref<1000xf32>
  }
  func @main(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x1000xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64x64x3x3xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64xf32>, %arg12: memref<64x64x3x3xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64xf32>, %arg17: memref<64x64x3x3xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64xf32>, %arg22: memref<64x64x3x3xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<64xf32>, %arg27: memref<128x64x3x3xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128xf32>, %arg32: memref<128x128x3x3xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128x64x1x1xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128x3x3xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128xf32>, %arg47: memref<128x128x3x3xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<256x128x3x3xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256xf32>, %arg57: memref<256x256x3x3xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256x128x1x1xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256x256x3x3xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256xf32>, %arg72: memref<256x256x3x3xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<256xf32>, %arg77: memref<512x256x3x3xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512xf32>, %arg82: memref<512x512x3x3xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512x256x1x1xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512xf32>, %arg91: memref<512xf32>, %arg92: memref<512x512x3x3xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512xf32>, %arg97: memref<512x512x3x3xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<512xf32>, %arg102: memref<1000x512xf32>, %arg103: memref<1000xf32>) -> (memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>) {
    %0 = memref.alloc() : memref<f32>
    %1 = memref.alloc() : memref<1000xf32>
    %2 = memref.alloc() : memref<f32>
    %3 = memref.alloc() : memref<4x64x112x112xf16>
    %4 = memref.alloc() : memref<4x512xf16>
    %5 = memref.alloc() : memref<4xf16>
    %6 = memref.alloc() : memref<4xf16>
    %7 = memref.alloc() : memref<4x1000xf16>
    %8 = memref.alloc() : memref<4x512xf16>
    %9 = memref.alloc() : memref<4x512x7x7xf16>
    %10 = memref.alloc() : memref<4x512x7x7xf16>
    %11 = memref.alloc() : memref<4x512x7x7xf16>
    %12 = memref.alloc() : memref<4x512x7x7xf16>
    %13 = memref.alloc() : memref<4x512x7x7xf16>
    %14 = memref.alloc() : memref<4x256x14x14xf16>
    %15 = memref.alloc() : memref<4x256x14x14xf16>
    %16 = memref.alloc() : memref<4x256x14x14xf16>
    %17 = memref.alloc() : memref<4x256x14x14xf16>
    %18 = memref.alloc() : memref<4x256x14x14xf16>
    %19 = memref.alloc() : memref<4x128x28x28xf16>
    %20 = memref.alloc() : memref<4x128x28x28xf16>
    %21 = memref.alloc() : memref<4x128x28x28xf16>
    %22 = memref.alloc() : memref<4x128x28x28xf16>
    %23 = memref.alloc() : memref<4x128x28x28xf16>
    %24 = memref.alloc() : memref<4x64x56x56xf16>
    %25 = memref.alloc() : memref<4x64x56x56xf16>
    %26 = memref.alloc() : memref<4x64x56x56xf16>
    %27 = memref.alloc() : memref<4x64x56x56xf16>
    %28 = memref.alloc() : memref<4x64x56x56xf16>
    %29 = memref.alloc() : memref<4xf16>
    %30 = memref.alloc() : memref<4x64x112x112xf16>
    %31 = memref.alloc() : memref<f16>
    %32 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%32) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.constant"(%31) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %33 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16>
    %34 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    lmhlo.convolution(%33, %34, %30) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>) -> ()
    %35 = call @BatchNormTrainingOp2(%30, %arg3, %arg4) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x112x112xf16>
    %36 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %37 = call @Unknown4(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %38 = call @Unknown5(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %39 = call @Unknown6(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %40 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %41 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %42 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %43 = call @Unknown10(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %44 = call @Unknown11(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %45 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %46 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %47 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %48 = call @Unknown15(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %49 = call @Unknown16(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %50 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %51 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %52 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %53 = call @Unknown20(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %54 = call @Unknown21(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %55 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %56 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %57 = call @Unknown24(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    "lmhlo.reduce"(%55, %32, %29) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %58:2 = call @Unknown25(%35) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    "lmhlo.reduce_window"(%58#0, %31, %28) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      %200 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %200) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%200, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<f16>, memref<4x64x56x56xf16>) -> ()
    lmhlo.convolution(%28, %36, %27) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %59 = call @BatchNormTrainingOp26(%27, %arg8, %arg9) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %60:2 = call @Unknown27(%59) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    lmhlo.convolution(%60#0, %37, %26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %61 = call @BatchNormTrainingOp28(%26, %arg13, %arg14) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %62:2 = call @Unknown29(%61, %28) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    lmhlo.convolution(%62#0, %38, %25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %63 = call @BatchNormTrainingOp30(%25, %arg18, %arg19) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %64:2 = call @Unknown31(%63) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    lmhlo.convolution(%64#0, %39, %24) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %65 = call @BatchNormTrainingOp32(%24, %arg23, %arg24) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %66:2 = call @Unknown33(%65, %62#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    lmhlo.convolution(%66#0, %40, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>) -> ()
    %67 = call @BatchNormTrainingOp34(%23, %arg38, %arg39) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    lmhlo.convolution(%66#0, %41, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %68 = call @BatchNormTrainingOp35(%22, %arg28, %arg29) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %69:2 = call @Unknown36(%68) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    lmhlo.convolution(%69#0, %42, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %70 = call @BatchNormTrainingOp37(%21, %arg33, %arg34) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %71:2 = call @Unknown38(%70, %67) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    lmhlo.convolution(%71#0, %43, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %72 = call @BatchNormTrainingOp39(%20, %arg43, %arg44) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %73:2 = call @Unknown40(%72) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    lmhlo.convolution(%73#0, %44, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %74 = call @BatchNormTrainingOp41(%19, %arg48, %arg49) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %75:2 = call @Unknown42(%74, %71#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    lmhlo.convolution(%75#0, %45, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>) -> ()
    %76 = call @BatchNormTrainingOp43(%18, %arg63, %arg64) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    lmhlo.convolution(%75#0, %46, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %77 = call @BatchNormTrainingOp44(%17, %arg53, %arg54) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %78:2 = call @Unknown45(%77) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    lmhlo.convolution(%78#0, %47, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %79 = call @BatchNormTrainingOp46(%16, %arg58, %arg59) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %80:2 = call @Unknown47(%79, %76) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    lmhlo.convolution(%80#0, %48, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %81 = call @BatchNormTrainingOp48(%15, %arg68, %arg69) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %82:2 = call @Unknown49(%81) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    lmhlo.convolution(%82#0, %49, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %83 = call @BatchNormTrainingOp50(%14, %arg73, %arg74) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %84:2 = call @Unknown51(%83, %80#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    lmhlo.convolution(%84#0, %50, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>) -> ()
    %85 = call @BatchNormTrainingOp52(%13, %arg88, %arg89) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    lmhlo.convolution(%84#0, %51, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %86 = call @BatchNormTrainingOp53(%12, %arg78, %arg79) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %87:2 = call @Unknown54(%86) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    lmhlo.convolution(%87#0, %52, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %88 = call @BatchNormTrainingOp55(%11, %arg83, %arg84) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %89:2 = call @Unknown56(%88, %85) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    lmhlo.convolution(%89#0, %53, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %90 = call @BatchNormTrainingOp57(%10, %arg93, %arg94) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %91:2 = call @Unknown58(%90) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    lmhlo.convolution(%91#0, %54, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %92 = call @BatchNormTrainingOp59(%9, %arg98, %arg99) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %93:2 = call @Unknown60(%92, %89#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    "lmhlo.reduce"(%93#0, %32, %8) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<4x512x7x7xf16>, memref<f16>, memref<4x512xf16>) -> ()
    %94 = call @Unknown61(%8) : (memref<4x512xf16>) -> memref<4x512xf16>
    "lmhlo.dot"(%94, %56, %7) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>) -> ()
    %95 = call @Unknown62(%57, %7) : (memref<1000xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    "lmhlo.reduce"(%95, %31, %6) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.maximum"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %96:2 = call @Unknown63(%6, %95) : (memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    "lmhlo.reduce"(%96#1, %32, %5) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %97 = call @Unknown64(%5) : (memref<4xf16>) -> memref<4xf16>
    %98:3 = call @Unknown65(%97, %96#0, %29, %55, %arg1) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>)
    "lmhlo.dot"(%98#0, %56, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>) -> ()
    %99 = call @Unknown66(%4, %93#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %100:3 = call @BatchNormGradOp67(%9, %arg98, %99) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %101 = call @ConvBackwardDataOp68(%100#0, %54) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %102 = call @ConvBackwardFilterOp69(%91#0, %100#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %103 = call @Unknown70(%91#1, %101) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %104:3 = call @BatchNormGradOp71(%10, %arg93, %103) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %105 = call @ConvBackwardDataOp72(%104#0, %53) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %106 = call @ConvBackwardFilterOp73(%89#0, %104#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %107 = call @Unknown74(%99, %105, %89#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %108:3 = call @BatchNormGradOp75(%11, %arg83, %107) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %109 = call @ConvBackwardDataOp76(%108#0, %52) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %110 = call @ConvBackwardFilterOp77(%87#0, %108#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %111 = call @Unknown78(%87#1, %109) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %112:3 = call @BatchNormGradOp79(%12, %arg78, %111) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %113 = call @ConvBackwardDataOp80(%112#0, %51) : (memref<4x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %114 = call @ConvBackwardFilterOp81(%84#0, %112#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %115:3 = call @BatchNormGradOp82(%13, %arg88, %107) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %116 = call @ConvBackwardDataOp83(%115#0, %50) : (memref<4x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16>
    %117 = call @ConvBackwardFilterOp84(%84#0, %115#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %118 = call @Unknown85(%116, %113, %84#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %119:3 = call @BatchNormGradOp86(%14, %arg73, %118) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %120 = call @ConvBackwardDataOp87(%119#0, %49) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %121 = call @ConvBackwardFilterOp88(%82#0, %119#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %122 = call @Unknown89(%82#1, %120) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %123:3 = call @BatchNormGradOp90(%15, %arg68, %122) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %124 = call @ConvBackwardDataOp91(%123#0, %48) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %125 = call @ConvBackwardFilterOp92(%80#0, %123#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %126 = call @Unknown93(%118, %124, %80#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %127:3 = call @BatchNormGradOp94(%16, %arg58, %126) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %128 = call @ConvBackwardDataOp95(%127#0, %47) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %129 = call @ConvBackwardFilterOp96(%78#0, %127#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %130 = call @Unknown97(%78#1, %128) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %131:3 = call @BatchNormGradOp98(%17, %arg53, %130) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %132 = call @ConvBackwardDataOp99(%131#0, %46) : (memref<4x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %133 = call @ConvBackwardFilterOp100(%75#0, %131#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %134:3 = call @BatchNormGradOp101(%18, %arg63, %126) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %135 = call @ConvBackwardDataOp102(%134#0, %45) : (memref<4x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16>
    %136 = call @ConvBackwardFilterOp103(%75#0, %134#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %137 = call @Unknown104(%135, %132, %75#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %138:3 = call @BatchNormGradOp105(%19, %arg48, %137) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %139 = call @ConvBackwardDataOp106(%138#0, %44) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %140 = call @ConvBackwardFilterOp107(%73#0, %138#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %141 = call @Unknown108(%73#1, %139) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %142:3 = call @BatchNormGradOp109(%20, %arg43, %141) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %143 = call @ConvBackwardDataOp110(%142#0, %43) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %144 = call @ConvBackwardFilterOp111(%71#0, %142#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %145 = call @Unknown112(%137, %143, %71#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %146:3 = call @BatchNormGradOp113(%21, %arg33, %145) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %147 = call @ConvBackwardDataOp114(%146#0, %42) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %148 = call @ConvBackwardFilterOp115(%69#0, %146#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %149 = call @Unknown116(%69#1, %147) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %150:3 = call @BatchNormGradOp117(%22, %arg28, %149) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %151 = call @ConvBackwardDataOp118(%150#0, %41) : (memref<4x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %152 = call @ConvBackwardFilterOp119(%66#0, %150#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %153:3 = call @BatchNormGradOp120(%23, %arg38, %145) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %154 = call @ConvBackwardDataOp121(%153#0, %40) : (memref<4x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp122(%66#0, %153#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %156 = call @Unknown123(%154, %151, %66#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %157:3 = call @BatchNormGradOp124(%24, %arg23, %156) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %158 = call @ConvBackwardDataOp125(%157#0, %39) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %159 = call @ConvBackwardFilterOp126(%64#0, %157#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %160 = call @Unknown127(%64#1, %158) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %161:3 = call @BatchNormGradOp128(%25, %arg18, %160) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %162 = call @ConvBackwardDataOp129(%161#0, %38) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %163 = call @ConvBackwardFilterOp130(%62#0, %161#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %164 = call @Unknown131(%156, %162, %62#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %165:3 = call @BatchNormGradOp132(%26, %arg13, %164) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %166 = call @ConvBackwardDataOp133(%165#0, %37) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %167 = call @ConvBackwardFilterOp134(%60#0, %165#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %168 = call @Unknown135(%60#1, %166) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %169:3 = call @BatchNormGradOp136(%27, %arg8, %168) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %170 = call @ConvBackwardDataOp137(%169#0, %36) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %171 = call @ConvBackwardFilterOp138(%28, %169#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %172 = call @Unknown139(%164, %170) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    "lmhlo.select_and_scatter"(%58#0, %172, %32, %3) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %200 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%200) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):
      %200 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%200) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<f16>, memref<4x64x112x112xf16>) -> ()
    %173 = call @Unknown140(%58#1, %3) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %174:3 = call @BatchNormGradOp141(%30, %arg3, %173) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %175 = call @ConvBackwardFilterOp142(%33, %174#0) : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16>
    "lmhlo.reduce"(%98#1, %0, %2) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<4x1000xf32>, memref<f32>, memref<f32>) -> ()
    %176 = call @Unknown143(%2) : (memref<f32>) -> memref<f32>
    %177 = call @Unknown144(%175) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %178 = call @Unknown145(%171) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %179 = call @Unknown146(%167) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %180 = call @Unknown147(%163) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %181 = call @Unknown148(%159) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %182 = call @Unknown149(%152) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %183 = call @Unknown150(%148) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %184 = call @Unknown151(%155) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %185 = call @Unknown152(%144) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %186 = call @Unknown153(%140) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %187 = call @Unknown154(%133) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %188 = call @Unknown155(%129) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %189 = call @Unknown156(%136) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %190 = call @Unknown157(%125) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %191 = call @Unknown158(%121) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %192 = call @Unknown159(%114) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %193 = call @Unknown160(%110) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %194 = call @Unknown161(%117) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %195 = call @Unknown162(%106) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %196 = call @Unknown163(%102) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %197 = call @MatmulOp164(%94, %98#0) : (memref<4x512xf16>, memref<4x1000xf16>) -> memref<1000x512xf16>
    %198 = call @Unknown165(%197) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    "lmhlo.reduce"(%98#2, %0, %1) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<4x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %199 = call @Unknown166(%1) : (memref<1000xf32>) -> memref<1000xf32>
    return %176, %177, %174#1, %174#2, %178, %169#1, %169#2, %179, %165#1, %165#2, %180, %161#1, %161#2, %181, %157#1, %157#2, %182, %150#1, %150#2, %183, %146#1, %146#2, %184, %153#1, %153#2, %185, %142#1, %142#2, %186, %138#1, %138#2, %187, %131#1, %131#2, %188, %127#1, %127#2, %189, %134#1, %134#2, %190, %123#1, %123#2, %191, %119#1, %119#2, %192, %112#1, %112#2, %193, %108#1, %108#2, %194, %115#1, %115#2, %195, %104#1, %104#2, %196, %100#1, %100#2, %198, %199 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}

