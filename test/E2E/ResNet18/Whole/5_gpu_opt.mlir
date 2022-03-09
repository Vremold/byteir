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
#map8 = affine_map<(d0) -> (d0 mod 512)>
#map9 = affine_map<(d0) -> (d0 floordiv 512)>
#map10 = affine_map<(d0) -> (d0 mod 3)>
#map11 = affine_map<(d0) -> ((d0 floordiv 3) mod 3)>
#map12 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 64)>
#map13 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 64)>
#map14 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 128)>
#map15 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 128)>
#map16 = affine_map<(d0) -> (d0 mod 64)>
#map17 = affine_map<(d0) -> (d0 floordiv 64)>
#map18 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 256)>
#map19 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 256)>
#map20 = affine_map<(d0) -> (d0 mod 128)>
#map21 = affine_map<(d0) -> (d0 floordiv 128)>
#map22 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) mod 512)>
#map23 = affine_map<(d0) -> (((d0 floordiv 3) floordiv 3) floordiv 512)>
#map24 = affine_map<(d0) -> (d0 mod 256)>
#map25 = affine_map<(d0) -> (d0 floordiv 256)>
#map26 = affine_map<(d0) -> (d0 mod 112)>
#map27 = affine_map<(d0) -> ((d0 floordiv 112) mod 112)>
#map28 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) mod 64)>
#map29 = affine_map<(d0) -> (((d0 floordiv 112) floordiv 112) floordiv 64)>
#map30 = affine_map<(d0) -> (d0 mod 56)>
#map31 = affine_map<(d0) -> ((d0 floordiv 56) mod 56)>
#map32 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) mod 64)>
#map33 = affine_map<(d0) -> (((d0 floordiv 56) floordiv 56) floordiv 64)>
#map34 = affine_map<(d0) -> (d0 mod 28)>
#map35 = affine_map<(d0) -> ((d0 floordiv 28) mod 28)>
#map36 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) mod 128)>
#map37 = affine_map<(d0) -> (((d0 floordiv 28) floordiv 28) floordiv 128)>
#map38 = affine_map<(d0) -> (d0 mod 14)>
#map39 = affine_map<(d0) -> ((d0 floordiv 14) mod 14)>
#map40 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) mod 256)>
#map41 = affine_map<(d0) -> (((d0 floordiv 14) floordiv 14) floordiv 256)>
#map42 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) mod 512)>
#map43 = affine_map<(d0) -> (((d0 floordiv 7) floordiv 7) floordiv 512)>
#map44 = affine_map<(d0) -> (d0 mod 1000)>
#map45 = affine_map<(d0) -> (d0 floordiv 1000)>
module {
  func private @Unknown0(%arg0: memref<32x3x224x224xf32>) -> memref<32x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<32x3x224x224xf16>
    affine.for %arg1 = 0 to 4816896 {
      %1 = affine.apply #map0(%arg1)
      %2 = affine.apply #map1(%arg1)
      %3 = affine.apply #map2(%arg1)
      %4 = affine.apply #map3(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x3x224x224xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<32x3x224x224xf16>
    }
    return %0 : memref<32x3x224x224xf16>
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
  func private @BatchNormTrainingOp2(%arg0: memref<32x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x112x112xf32>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x112x112xf32>, memref<32x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown3(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf16>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1] : memref<1000x512xf16>
    }
    return %0 : memref<1000x512xf16>
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown7(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf16>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
    }
    return %0 : memref<128x64x3x3xf16>
  }
  func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown10(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<128x64x1x1xf16>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
    }
    return %0 : memref<128x64x1x1xf16>
  }
  func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown12(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf16>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
    }
    return %0 : memref<256x128x3x3xf16>
  }
  func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown15(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<256x128x1x1xf16>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map20(%arg1)
      %2 = affine.apply #map21(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
    }
    return %0 : memref<256x128x1x1xf16>
  }
  func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown17(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf16>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
    }
    return %0 : memref<512x256x3x3xf16>
  }
  func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown20(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<512x256x1x1xf16>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map24(%arg1)
      %2 = affine.apply #map25(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
      %4 = arith.truncf %3 : f32 to f16
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
    }
    return %0 : memref<512x256x1x1xf16>
  }
  func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
      %6 = arith.truncf %5 : f32 to f16
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown23(%arg0: memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<32x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x112x112xi1>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    affine.for %arg1 = 0 to 25690112 {
      %2 = affine.apply #map26(%arg1)
      %3 = affine.apply #map27(%arg1)
      %4 = affine.apply #map28(%arg1)
      %5 = affine.apply #map29(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x64x112x112xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x64x112x112xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x64x112x112xi1>
    }
    return %1, %0 : memref<32x64x112x112xf16>, memref<32x64x112x112xi1>
  }
  func private @BatchNormTrainingOp24(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown25(%arg0: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg1 = 0 to 6422528 {
      %2 = affine.apply #map30(%arg1)
      %3 = affine.apply #map31(%arg1)
      %4 = affine.apply #map32(%arg1)
      %5 = affine.apply #map33(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x64x56x56xi1>
    }
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp26(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown27(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg2 = 0 to 6422528 {
      %2 = affine.apply #map30(%arg2)
      %3 = affine.apply #map31(%arg2)
      %4 = affine.apply #map32(%arg2)
      %5 = affine.apply #map33(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x64x56x56xi1>
    }
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp28(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown29(%arg0: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg1 = 0 to 6422528 {
      %2 = affine.apply #map30(%arg1)
      %3 = affine.apply #map31(%arg1)
      %4 = affine.apply #map32(%arg1)
      %5 = affine.apply #map33(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x64x56x56xi1>
    }
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp30(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @Unknown31(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xi1>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg2 = 0 to 6422528 {
      %2 = affine.apply #map30(%arg2)
      %3 = affine.apply #map31(%arg2)
      %4 = affine.apply #map32(%arg2)
      %5 = affine.apply #map33(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x64x56x56xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x64x56x56xi1>
    }
    return %1, %0 : memref<32x64x56x56xf16>, memref<32x64x56x56xi1>
  }
  func private @BatchNormTrainingOp32(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp33(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown34(%arg0: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg1 = 0 to 3211264 {
      %2 = affine.apply #map34(%arg1)
      %3 = affine.apply #map35(%arg1)
      %4 = affine.apply #map36(%arg1)
      %5 = affine.apply #map37(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x128x28x28xi1>
    }
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp35(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown36(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg2 = 0 to 3211264 {
      %2 = affine.apply #map34(%arg2)
      %3 = affine.apply #map35(%arg2)
      %4 = affine.apply #map36(%arg2)
      %5 = affine.apply #map37(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x128x28x28xi1>
    }
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp37(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown38(%arg0: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg1 = 0 to 3211264 {
      %2 = affine.apply #map34(%arg1)
      %3 = affine.apply #map35(%arg1)
      %4 = affine.apply #map36(%arg1)
      %5 = affine.apply #map37(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x128x28x28xi1>
    }
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp39(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @Unknown40(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xi1>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg2 = 0 to 3211264 {
      %2 = affine.apply #map34(%arg2)
      %3 = affine.apply #map35(%arg2)
      %4 = affine.apply #map36(%arg2)
      %5 = affine.apply #map37(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x128x28x28xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x128x28x28xi1>
    }
    return %1, %0 : memref<32x128x28x28xf16>, memref<32x128x28x28xi1>
  }
  func private @BatchNormTrainingOp41(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp42(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown43(%arg0: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg1 = 0 to 1605632 {
      %2 = affine.apply #map38(%arg1)
      %3 = affine.apply #map39(%arg1)
      %4 = affine.apply #map40(%arg1)
      %5 = affine.apply #map41(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x256x14x14xi1>
    }
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp44(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown45(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg2 = 0 to 1605632 {
      %2 = affine.apply #map38(%arg2)
      %3 = affine.apply #map39(%arg2)
      %4 = affine.apply #map40(%arg2)
      %5 = affine.apply #map41(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x256x14x14xi1>
    }
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp46(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown47(%arg0: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg1 = 0 to 1605632 {
      %2 = affine.apply #map38(%arg1)
      %3 = affine.apply #map39(%arg1)
      %4 = affine.apply #map40(%arg1)
      %5 = affine.apply #map41(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x256x14x14xi1>
    }
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp48(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @Unknown49(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xi1>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg2 = 0 to 1605632 {
      %2 = affine.apply #map38(%arg2)
      %3 = affine.apply #map39(%arg2)
      %4 = affine.apply #map40(%arg2)
      %5 = affine.apply #map41(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x256x14x14xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x256x14x14xi1>
    }
    return %1, %0 : memref<32x256x14x14xf16>, memref<32x256x14x14xi1>
  }
  func private @BatchNormTrainingOp50(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @BatchNormTrainingOp51(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown52(%arg0: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg1 = 0 to 802816 {
      %2 = affine.apply #map4(%arg1)
      %3 = affine.apply #map5(%arg1)
      %4 = affine.apply #map42(%arg1)
      %5 = affine.apply #map43(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x512x7x7xi1>
    }
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp53(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown54(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg2 = 0 to 802816 {
      %2 = affine.apply #map4(%arg2)
      %3 = affine.apply #map5(%arg2)
      %4 = affine.apply #map42(%arg2)
      %5 = affine.apply #map43(%arg2)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %7 = affine.load %arg1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %10 = arith.addf %6, %7 : f16
      %11 = arith.maxf %10, %cst : f16
      %12 = arith.cmpf ogt, %11, %cst : f16
      affine.store %12, %0[%5, %4, %3, %2] : memref<32x512x7x7xi1>
    }
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp55(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown56(%arg0: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xi1>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg1 = 0 to 802816 {
      %2 = affine.apply #map4(%arg1)
      %3 = affine.apply #map5(%arg1)
      %4 = affine.apply #map42(%arg1)
      %5 = affine.apply #map43(%arg1)
      %6 = affine.load %arg0[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %7 = arith.maxf %6, %cst : f16
      affine.store %7, %1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %8 = arith.maxf %6, %cst : f16
      %9 = arith.cmpf ogt, %8, %cst : f16
      affine.store %9, %0[%5, %4, %3, %2] : memref<32x512x7x7xi1>
    }
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xi1>
  }
  func private @BatchNormTrainingOp57(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @Unknown58(%arg0: memref<32x512xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %cst_0 = arith.constant 4.900000e+01 : f16
    affine.for %arg3 = 0 to 802816 {
      %2 = affine.apply #map4(%arg3)
      %3 = affine.apply #map5(%arg3)
      %4 = affine.apply #map42(%arg3)
      %5 = affine.apply #map43(%arg3)
      %6 = affine.load %arg1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %7 = affine.load %arg2[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = arith.maxf %8, %cst : f16
      affine.store %9, %1[%5, %4, %3, %2] : memref<32x512x7x7xf16>
      %10 = affine.load %arg0[%5, %4] : memref<32x512xf16>
      %11 = arith.divf %10, %cst_0 : f16
      %12 = arith.addf %6, %7 : f16
      %13 = arith.maxf %12, %cst : f16
      %14 = arith.cmpf ogt, %13, %cst : f16
      %15 = select %14, %11, %cst : f16
      affine.store %15, %0[%5, %4, %3, %2] : memref<32x512x7x7xf16>
    }
    return %1, %0 : memref<32x512x7x7xf16>, memref<32x512x7x7xf16>
  }
  func private @Unknown59(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @BatchNormGradOp60(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp61(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp62(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown63(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map42(%arg2)
      %4 = affine.apply #map43(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x512x7x7xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x512x7x7xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x512x7x7xf16>
    }
    return %0 : memref<32x512x7x7xf16>
  }
  func private @Unknown64(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @BatchNormGradOp65(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp66(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp67(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown68(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg3 = 0 to 802816 {
      %1 = affine.apply #map4(%arg3)
      %2 = affine.apply #map5(%arg3)
      %3 = affine.apply #map42(%arg3)
      %4 = affine.apply #map43(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x512x7x7xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x512x7x7xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x512x7x7xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x512x7x7xf16>
    }
    return %0 : memref<32x512x7x7xf16>
  }
  func private @Unknown69(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @BatchNormGradOp70(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp71(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp72(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown73(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x512x7x7xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map42(%arg2)
      %4 = affine.apply #map43(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x512x7x7xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x512x7x7xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x512x7x7xf16>
    }
    return %0 : memref<32x512x7x7xf16>
  }
  func private @Unknown74(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @BatchNormGradOp75(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp76(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp77(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @Unknown78(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<512xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<512xf32>
    }
    return %0 : memref<512xf32>
  }
  func private @BatchNormGradOp79(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x512x7x7xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<1x1x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown82(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg3 = 0 to 1605632 {
      %1 = affine.apply #map38(%arg3)
      %2 = affine.apply #map39(%arg3)
      %3 = affine.apply #map40(%arg3)
      %4 = affine.apply #map41(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x256x14x14xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
    }
    return %0 : memref<32x256x14x14xf16>
  }
  func private @Unknown83(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @BatchNormGradOp84(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp85(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp86(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown87(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg2 = 0 to 1605632 {
      %1 = affine.apply #map38(%arg2)
      %2 = affine.apply #map39(%arg2)
      %3 = affine.apply #map40(%arg2)
      %4 = affine.apply #map41(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x256x14x14xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
    }
    return %0 : memref<32x256x14x14xf16>
  }
  func private @Unknown88(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @BatchNormGradOp89(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp90(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp91(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown92(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg3 = 0 to 1605632 {
      %1 = affine.apply #map38(%arg3)
      %2 = affine.apply #map39(%arg3)
      %3 = affine.apply #map40(%arg3)
      %4 = affine.apply #map41(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x256x14x14xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
    }
    return %0 : memref<32x256x14x14xf16>
  }
  func private @Unknown93(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @BatchNormGradOp94(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown97(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x256x14x14xf16>
    affine.for %arg2 = 0 to 1605632 {
      %1 = affine.apply #map38(%arg2)
      %2 = affine.apply #map39(%arg2)
      %3 = affine.apply #map40(%arg2)
      %4 = affine.apply #map41(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x256x14x14xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x256x14x14xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x256x14x14xf16>
    }
    return %0 : memref<32x256x14x14xf16>
  }
  func private @Unknown98(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @BatchNormGradOp99(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp100(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp101(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @Unknown102(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<256xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 256 {
      %1 = affine.load %arg0[%arg1] : memref<256xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<256xf32>
    }
    return %0 : memref<256xf32>
  }
  func private @BatchNormGradOp103(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x256x14x14xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp104(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<1x1x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp105(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown106(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg3 = 0 to 3211264 {
      %1 = affine.apply #map34(%arg3)
      %2 = affine.apply #map35(%arg3)
      %3 = affine.apply #map36(%arg3)
      %4 = affine.apply #map37(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x128x28x28xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
    }
    return %0 : memref<32x128x28x28xf16>
  }
  func private @Unknown107(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 128 {
      %1 = affine.load %arg0[%arg1] : memref<128xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @BatchNormGradOp108(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp109(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp110(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown111(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg2 = 0 to 3211264 {
      %1 = affine.apply #map34(%arg2)
      %2 = affine.apply #map35(%arg2)
      %3 = affine.apply #map36(%arg2)
      %4 = affine.apply #map37(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x128x28x28xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
    }
    return %0 : memref<32x128x28x28xf16>
  }
  func private @Unknown112(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 128 {
      %1 = affine.load %arg0[%arg1] : memref<128xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @BatchNormGradOp113(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp114(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp115(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown116(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg3 = 0 to 3211264 {
      %1 = affine.apply #map34(%arg3)
      %2 = affine.apply #map35(%arg3)
      %3 = affine.apply #map36(%arg3)
      %4 = affine.apply #map37(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x128x28x28xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
    }
    return %0 : memref<32x128x28x28xf16>
  }
  func private @Unknown117(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 128 {
      %1 = affine.load %arg0[%arg1] : memref<128xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @BatchNormGradOp118(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp119(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp120(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown121(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x128x28x28xf16>
    affine.for %arg2 = 0 to 3211264 {
      %1 = affine.apply #map34(%arg2)
      %2 = affine.apply #map35(%arg2)
      %3 = affine.apply #map36(%arg2)
      %4 = affine.apply #map37(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x128x28x28xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x128x28x28xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x128x28x28xf16>
    }
    return %0 : memref<32x128x28x28xf16>
  }
  func private @Unknown122(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 128 {
      %1 = affine.load %arg0[%arg1] : memref<128xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @BatchNormGradOp123(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp124(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp125(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @Unknown126(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<128xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 128 {
      %1 = affine.load %arg0[%arg1] : memref<128xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<128xf32>
    }
    return %0 : memref<128xf32>
  }
  func private @BatchNormGradOp127(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x128x28x28xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp128(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<1x1x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp129(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown130(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg3 = 0 to 6422528 {
      %1 = affine.apply #map30(%arg3)
      %2 = affine.apply #map31(%arg3)
      %3 = affine.apply #map32(%arg3)
      %4 = affine.apply #map33(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x64x56x56xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
    }
    return %0 : memref<32x64x56x56xf16>
  }
  func private @Unknown131(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 64 {
      %1 = affine.load %arg0[%arg1] : memref<64xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @BatchNormGradOp132(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp133(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp134(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown135(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg2 = 0 to 6422528 {
      %1 = affine.apply #map30(%arg2)
      %2 = affine.apply #map31(%arg2)
      %3 = affine.apply #map32(%arg2)
      %4 = affine.apply #map33(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x56x56xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
    }
    return %0 : memref<32x64x56x56xf16>
  }
  func private @Unknown136(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 64 {
      %1 = affine.load %arg0[%arg1] : memref<64xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @BatchNormGradOp137(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp138(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp139(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown140(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg3 = 0 to 6422528 {
      %1 = affine.apply #map30(%arg3)
      %2 = affine.apply #map31(%arg3)
      %3 = affine.apply #map32(%arg3)
      %4 = affine.apply #map33(%arg3)
      %5 = affine.load %arg2[%4, %3, %2, %1] : memref<32x64x56x56xi1>
      %6 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %7 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %8 = arith.addf %6, %7 : f16
      %9 = select %5, %8, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
    }
    return %0 : memref<32x64x56x56xf16>
  }
  func private @Unknown141(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 64 {
      %1 = affine.load %arg0[%arg1] : memref<64xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @BatchNormGradOp142(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp143(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp144(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown145(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg2 = 0 to 6422528 {
      %1 = affine.apply #map30(%arg2)
      %2 = affine.apply #map31(%arg2)
      %3 = affine.apply #map32(%arg2)
      %4 = affine.apply #map33(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x56x56xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
    }
    return %0 : memref<32x64x56x56xf16>
  }
  func private @Unknown146(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 64 {
      %1 = affine.load %arg0[%arg1] : memref<64xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @BatchNormGradOp147(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x56x56xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp148(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp149(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown150(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<32x64x56x56xf16>
    affine.for %arg2 = 0 to 6422528 {
      %1 = affine.apply #map30(%arg2)
      %2 = affine.apply #map31(%arg2)
      %3 = affine.apply #map32(%arg2)
      %4 = affine.apply #map33(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x56x56xf16>
      %7 = arith.addf %5, %6 : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x64x56x56xf16>
    }
    return %0 : memref<32x64x56x56xf16>
  }
  func private @Unknown151(%arg0: memref<32x64x112x112xi1>, %arg1: memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<32x64x112x112xf16>
    affine.for %arg2 = 0 to 25690112 {
      %1 = affine.apply #map26(%arg2)
      %2 = affine.apply #map27(%arg2)
      %3 = affine.apply #map28(%arg2)
      %4 = affine.apply #map29(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<32x64x112x112xi1>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<32x64x112x112xf16>
      %7 = select %5, %6, %cst : f16
      affine.store %7, %0[%4, %3, %2, %1] : memref<32x64x112x112xf16>
    }
    return %0 : memref<32x64x112x112xf16>
  }
  func private @Unknown152(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 9.99999974E-6 : f32
    %0 = memref.alloc() : memref<64xf32>
    %cst_0 = arith.constant 1.000000e+00 : f32
    affine.for %arg1 = 0 to 64 {
      %1 = affine.load %arg0[%arg1] : memref<64xf32>
      %2 = arith.addf %1, %cst : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.divf %cst_0, %3 : f32
      %5 = arith.mulf %4, %4 : f32
      %6 = arith.subf %5, %cst : f32
      affine.store %6, %0[%arg1] : memref<64xf32>
    }
    return %0 : memref<64xf32>
  }
  func private @BatchNormGradOp153(%arg0: memref<32x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<32x64x112x112xf32>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x112x112xf32>
    %5 = memref.alloc() : memref<32x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf32>, memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x112x112xf32>, memref<32x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp154(%arg0: memref<32x3x224x224xf16>, %arg1: memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown155(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x3x7x7xf32>
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
  func private @Unknown156(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown157(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown158(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown159(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    affine.for %arg1 = 0 to 36864 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<64x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<64x64x3x3xf32>
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown160(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf32>
    affine.for %arg1 = 0 to 73728 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map12(%arg1)
      %4 = affine.apply #map13(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x64x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x64x3x3xf32>
    }
    return %0 : memref<128x64x3x3xf32>
  }
  func private @Unknown161(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown162(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<128x64x1x1xf32>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map16(%arg1)
      %2 = affine.apply #map17(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func private @Unknown163(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown164(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    affine.for %arg1 = 0 to 147456 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<128x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<128x128x3x3xf32>
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown165(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf32>
    affine.for %arg1 = 0 to 294912 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map14(%arg1)
      %4 = affine.apply #map15(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x128x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x128x3x3xf32>
    }
    return %0 : memref<256x128x3x3xf32>
  }
  func private @Unknown166(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown167(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<256x128x1x1xf32>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map20(%arg1)
      %2 = affine.apply #map21(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func private @Unknown168(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown169(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    affine.for %arg1 = 0 to 589824 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<256x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<256x256x3x3xf32>
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown170(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf32>
    affine.for %arg1 = 0 to 1179648 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map18(%arg1)
      %4 = affine.apply #map19(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x256x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x256x3x3xf32>
    }
    return %0 : memref<512x256x3x3xf32>
  }
  func private @Unknown171(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown172(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<512x256x1x1xf32>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map24(%arg1)
      %2 = affine.apply #map25(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func private @Unknown173(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown174(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    affine.for %arg1 = 0 to 2359296 {
      %1 = affine.apply #map10(%arg1)
      %2 = affine.apply #map11(%arg1)
      %3 = affine.apply #map22(%arg1)
      %4 = affine.apply #map23(%arg1)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<512x512x3x3xf16>
      %6 = arith.extf %5 : f16 to f32
      affine.store %6, %0[%4, %3, %2, %1] : memref<512x512x3x3xf32>
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown175(%arg0: memref<32x512xf16>) -> memref<32x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = memref.alloc() : memref<32x512xf16>
    affine.for %arg1 = 0 to 16384 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<32x512xf16>
      %4 = arith.mulf %3, %cst : f16
      affine.store %4, %0[%2, %1] : memref<32x512xf16>
    }
    return %0 : memref<32x512xf16>
  }
  func private @MatmulOp176(%arg0: memref<32x512xf16>, %arg1: memref<32x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<512x1000xf16>
    %1 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<32x1000xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %1 : memref<1000x512xf16>
  }
  func private @Unknown177(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf32>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map8(%arg1)
      %2 = affine.apply #map9(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1000x512xf32>
    }
    return %0 : memref<1000x512xf32>
  }
  func private @Unknown178(%arg0: memref<32x1000xf16>) -> memref<32x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<32x1000xf32>
    affine.for %arg1 = 0 to 32000 {
      %1 = affine.apply #map44(%arg1)
      %2 = affine.apply #map45(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<32x1000xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<32x1000xf32>
    }
    return %0 : memref<32x1000xf32>
  }
  func private @Unknown179(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      affine.store %3, %0[%arg1] : memref<1000xf32>
    }
    return %0 : memref<1000xf32>
  }
  func private @Unknown180(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown181(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown182(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown183(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown184(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown185(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown186(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown187(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown188(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown189(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown190(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown191(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown192(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown193(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown194(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown195(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown196(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown197(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown198(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown199(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown200(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown201(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown202(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown203(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown204(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown205(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown206(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown207(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown208(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown209(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown210(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown211(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown212(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown213(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown214(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown215(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown216(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown217(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown218(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown219(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown220(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf16>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      affine.store %2, %0[%arg1] : memref<1000xf16>
    }
    return %0 : memref<1000xf16>
  }
  func private @Unknown221(%arg0: memref<1000xf16>, %arg1: memref<32x1000xf16>) -> memref<32x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<32x1000xf16>
    affine.for %arg2 = 0 to 32000 {
      %1 = affine.apply #map44(%arg2)
      %2 = affine.apply #map45(%arg2)
      %3 = affine.load %arg1[%2, %1] : memref<32x1000xf16>
      %4 = affine.load %arg0[%1] : memref<1000xf16>
      %5 = arith.addf %3, %4 : f16
      affine.store %5, %0[%2, %1] : memref<32x1000xf16>
    }
    return %0 : memref<32x1000xf16>
  }
  func @main(%arg0: memref<64x3x7x7xf32>, %arg1: memref<32x3x224x224xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64x64x3x3xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64x64x3x3xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<64xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64x64x3x3xf32>, %arg22: memref<64xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<128x64x3x3xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128x64x1x1xf32>, %arg41: memref<128x128x3x3xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128x128x3x3xf32>, %arg47: memref<128xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<256x128x3x3xf32>, %arg52: memref<256xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256x256x3x3xf32>, %arg57: memref<256xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256x128x1x1xf32>, %arg66: memref<256x256x3x3xf32>, %arg67: memref<256xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256x256x3x3xf32>, %arg72: memref<256xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<512x256x3x3xf32>, %arg77: memref<512xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512x512x3x3xf32>, %arg82: memref<512xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512x256x1x1xf32>, %arg91: memref<512x512x3x3xf32>, %arg92: memref<512xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512x512x3x3xf32>, %arg97: memref<512xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<1000x512xf32>, %arg102: memref<32x1000xf16>, %arg103: memref<1000xf32>) -> (memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<32x1000xf16>) {
    %0 = memref.alloc() : memref<i64>
    %1 = memref.alloc() : memref<32x1000xf16>
    %2 = memref.alloc() : memref<1000xf32>
    %3 = memref.alloc() : memref<32x512xf16>
    %4 = memref.alloc() : memref<32x64x112x112xf16>
    %5 = memref.alloc() : memref<32x512x7x7xf16>
    %6 = memref.alloc() : memref<32x512x7x7xf16>
    %7 = memref.alloc() : memref<32x512x7x7xf16>
    %8 = memref.alloc() : memref<32x512x7x7xf16>
    %9 = memref.alloc() : memref<32x512x7x7xf16>
    %10 = memref.alloc() : memref<32x256x14x14xf16>
    %11 = memref.alloc() : memref<32x256x14x14xf16>
    %12 = memref.alloc() : memref<32x256x14x14xf16>
    %13 = memref.alloc() : memref<32x256x14x14xf16>
    %14 = memref.alloc() : memref<32x256x14x14xf16>
    %15 = memref.alloc() : memref<32x128x28x28xf16>
    %16 = memref.alloc() : memref<32x128x28x28xf16>
    %17 = memref.alloc() : memref<32x128x28x28xf16>
    %18 = memref.alloc() : memref<32x128x28x28xf16>
    %19 = memref.alloc() : memref<32x128x28x28xf16>
    %20 = memref.alloc() : memref<32x64x56x56xf16>
    %21 = memref.alloc() : memref<32x64x56x56xf16>
    %22 = memref.alloc() : memref<32x64x56x56xf16>
    %23 = memref.alloc() : memref<32x64x56x56xf16>
    %24 = memref.alloc() : memref<32x64x56x56xf16>
    %25 = memref.alloc() : memref<32x512xf16>
    %26 = memref.alloc() : memref<32x64x112x112xf16>
    %27 = memref.alloc() : memref<f16>
    %28 = memref.alloc() : memref<f16>
    %29 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<1> : tensor<i64>} : (memref<i64>) -> ()
    "lmhlo.constant"(%29) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.constant"(%28) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    "lmhlo.constant"(%27) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %30 = call @Unknown0(%arg1) : (memref<32x3x224x224xf32>) -> memref<32x3x224x224xf16>
    %31 = call @Unknown1(%arg0) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    lmhlo.convolution(%30, %31, %26) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x3x224x224xf16>, memref<64x3x7x7xf16>, memref<32x64x112x112xf16>) -> ()
    %32:3 = call @BatchNormTrainingOp2(%26, %arg5, %arg4) : (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %33 = call @Unknown3(%arg101) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    "lmhlo.dot"(%arg102, %33, %25) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x1000xf16>, memref<1000x512xf16>, memref<32x512xf16>) -> ()
    %34 = call @Unknown4(%arg6) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %35 = call @Unknown5(%arg11) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %36 = call @Unknown6(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %37 = call @Unknown7(%arg21) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %38 = call @Unknown8(%arg26) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %39 = call @Unknown9(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %40 = call @Unknown10(%arg40) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %41 = call @Unknown11(%arg41) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %42 = call @Unknown12(%arg46) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %43 = call @Unknown13(%arg51) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %44 = call @Unknown14(%arg56) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %45 = call @Unknown15(%arg65) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %46 = call @Unknown16(%arg66) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %47 = call @Unknown17(%arg71) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %48 = call @Unknown18(%arg76) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %49 = call @Unknown19(%arg81) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %50 = call @Unknown20(%arg90) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %51 = call @Unknown21(%arg91) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %52 = call @Unknown22(%arg96) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %53:2 = call @Unknown23(%32#0) : (memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<32x64x112x112xi1>)
    "lmhlo.reduce_window"(%53#0, %27, %24) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      %252 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %252) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%252, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<f16>, memref<32x64x56x56xf16>) -> ()
    lmhlo.convolution(%24, %34, %23) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %54:3 = call @BatchNormTrainingOp24(%23, %arg10, %arg9) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %55:2 = call @Unknown25(%54#0) : (memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%55#0, %35, %22) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %56:3 = call @BatchNormTrainingOp26(%22, %arg15, %arg14) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %57:2 = call @Unknown27(%56#0, %24) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%57#0, %36, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %58:3 = call @BatchNormTrainingOp28(%21, %arg20, %arg19) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %59:2 = call @Unknown29(%58#0) : (memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%59#0, %37, %20) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>, memref<32x64x56x56xf16>) -> ()
    %60:3 = call @BatchNormTrainingOp30(%20, %arg25, %arg24) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %61:2 = call @Unknown31(%60#0, %57#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<32x64x56x56xi1>)
    lmhlo.convolution(%61#0, %38, %19) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<128x64x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %62:3 = call @BatchNormTrainingOp32(%19, %arg30, %arg29) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    lmhlo.convolution(%61#0, %40, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<128x64x1x1xf16>, memref<32x128x28x28xf16>) -> ()
    %63:3 = call @BatchNormTrainingOp33(%18, %arg39, %arg38) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %64:2 = call @Unknown34(%62#0) : (memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%64#0, %39, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %65:3 = call @BatchNormTrainingOp35(%17, %arg35, %arg34) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %66:2 = call @Unknown36(%65#0, %63#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%66#0, %41, %16) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %67:3 = call @BatchNormTrainingOp37(%16, %arg45, %arg44) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %68:2 = call @Unknown38(%67#0) : (memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%68#0, %42, %15) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>, memref<32x128x28x28xf16>) -> ()
    %69:3 = call @BatchNormTrainingOp39(%15, %arg50, %arg49) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %70:2 = call @Unknown40(%69#0, %66#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<32x128x28x28xi1>)
    lmhlo.convolution(%70#0, %43, %14) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<256x128x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %71:3 = call @BatchNormTrainingOp41(%14, %arg55, %arg54) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    lmhlo.convolution(%70#0, %45, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<256x128x1x1xf16>, memref<32x256x14x14xf16>) -> ()
    %72:3 = call @BatchNormTrainingOp42(%13, %arg64, %arg63) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %73:2 = call @Unknown43(%71#0) : (memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%73#0, %44, %12) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %74:3 = call @BatchNormTrainingOp44(%12, %arg60, %arg59) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %75:2 = call @Unknown45(%74#0, %72#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%75#0, %46, %11) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %76:3 = call @BatchNormTrainingOp46(%11, %arg70, %arg69) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %77:2 = call @Unknown47(%76#0) : (memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%77#0, %47, %10) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>, memref<32x256x14x14xf16>) -> ()
    %78:3 = call @BatchNormTrainingOp48(%10, %arg75, %arg74) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %79:2 = call @Unknown49(%78#0, %75#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<32x256x14x14xi1>)
    lmhlo.convolution(%79#0, %48, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<512x256x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %80:3 = call @BatchNormTrainingOp50(%9, %arg80, %arg79) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    lmhlo.convolution(%79#0, %50, %8) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<512x256x1x1xf16>, memref<32x512x7x7xf16>) -> ()
    %81:3 = call @BatchNormTrainingOp51(%8, %arg89, %arg88) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %82:2 = call @Unknown52(%80#0) : (memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%82#0, %49, %7) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %83:3 = call @BatchNormTrainingOp53(%7, %arg85, %arg84) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %84:2 = call @Unknown54(%83#0, %81#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%84#0, %51, %6) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %85:3 = call @BatchNormTrainingOp55(%6, %arg95, %arg94) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %86:2 = call @Unknown56(%85#0) : (memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xi1>)
    lmhlo.convolution(%86#0, %52, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>, memref<32x512x7x7xf16>) -> ()
    %87:3 = call @BatchNormTrainingOp57(%5, %arg100, %arg99) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %88:2 = call @Unknown58(%25, %87#0, %84#0) : (memref<32x512xf16>, memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>)
    %89 = call @Unknown59(%87#2) : (memref<512xf32>) -> memref<512xf32>
    %90:3 = call @BatchNormGradOp60(%5, %arg100, %87#1, %89, %88#1) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %91 = call @ConvBackwardDataOp61(%90#0, %52) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %92 = call @ConvBackwardFilterOp62(%86#0, %90#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %93 = call @Unknown63(%86#1, %91) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %94 = call @Unknown64(%85#2) : (memref<512xf32>) -> memref<512xf32>
    %95:3 = call @BatchNormGradOp65(%6, %arg95, %85#1, %94, %93) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %96 = call @ConvBackwardDataOp66(%95#0, %51) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %97 = call @ConvBackwardFilterOp67(%84#0, %95#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %98 = call @Unknown68(%88#1, %96, %84#1) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16>
    %99 = call @Unknown69(%83#2) : (memref<512xf32>) -> memref<512xf32>
    %100:3 = call @BatchNormGradOp70(%7, %arg85, %83#1, %99, %98) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %101 = call @ConvBackwardDataOp71(%100#0, %49) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %102 = call @ConvBackwardFilterOp72(%82#0, %100#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %103 = call @Unknown73(%82#1, %101) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %104 = call @Unknown74(%80#2) : (memref<512xf32>) -> memref<512xf32>
    %105:3 = call @BatchNormGradOp75(%9, %arg80, %80#1, %104, %103) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %106 = call @ConvBackwardDataOp76(%105#0, %48) : (memref<32x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %107 = call @ConvBackwardFilterOp77(%79#0, %105#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %108 = call @Unknown78(%81#2) : (memref<512xf32>) -> memref<512xf32>
    %109:3 = call @BatchNormGradOp79(%8, %arg89, %81#1, %108, %98) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %110 = call @ConvBackwardDataOp80(%109#0, %50) : (memref<32x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16>
    %111 = call @ConvBackwardFilterOp81(%79#0, %109#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %112 = call @Unknown82(%110, %106, %79#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %113 = call @Unknown83(%78#2) : (memref<256xf32>) -> memref<256xf32>
    %114:3 = call @BatchNormGradOp84(%10, %arg75, %78#1, %113, %112) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %115 = call @ConvBackwardDataOp85(%114#0, %47) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %116 = call @ConvBackwardFilterOp86(%77#0, %114#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %117 = call @Unknown87(%77#1, %115) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %118 = call @Unknown88(%76#2) : (memref<256xf32>) -> memref<256xf32>
    %119:3 = call @BatchNormGradOp89(%11, %arg70, %76#1, %118, %117) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %120 = call @ConvBackwardDataOp90(%119#0, %46) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %121 = call @ConvBackwardFilterOp91(%75#0, %119#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %122 = call @Unknown92(%112, %120, %75#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %123 = call @Unknown93(%74#2) : (memref<256xf32>) -> memref<256xf32>
    %124:3 = call @BatchNormGradOp94(%12, %arg60, %74#1, %123, %122) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %125 = call @ConvBackwardDataOp95(%124#0, %44) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %126 = call @ConvBackwardFilterOp96(%73#0, %124#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %127 = call @Unknown97(%73#1, %125) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %128 = call @Unknown98(%71#2) : (memref<256xf32>) -> memref<256xf32>
    %129:3 = call @BatchNormGradOp99(%14, %arg55, %71#1, %128, %127) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %130 = call @ConvBackwardDataOp100(%129#0, %43) : (memref<32x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %131 = call @ConvBackwardFilterOp101(%70#0, %129#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %132 = call @Unknown102(%72#2) : (memref<256xf32>) -> memref<256xf32>
    %133:3 = call @BatchNormGradOp103(%13, %arg64, %72#1, %132, %122) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %134 = call @ConvBackwardDataOp104(%133#0, %45) : (memref<32x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16>
    %135 = call @ConvBackwardFilterOp105(%70#0, %133#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %136 = call @Unknown106(%134, %130, %70#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %137 = call @Unknown107(%69#2) : (memref<128xf32>) -> memref<128xf32>
    %138:3 = call @BatchNormGradOp108(%15, %arg50, %69#1, %137, %136) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %139 = call @ConvBackwardDataOp109(%138#0, %42) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %140 = call @ConvBackwardFilterOp110(%68#0, %138#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %141 = call @Unknown111(%68#1, %139) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %142 = call @Unknown112(%67#2) : (memref<128xf32>) -> memref<128xf32>
    %143:3 = call @BatchNormGradOp113(%16, %arg45, %67#1, %142, %141) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %144 = call @ConvBackwardDataOp114(%143#0, %41) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %145 = call @ConvBackwardFilterOp115(%66#0, %143#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %146 = call @Unknown116(%136, %144, %66#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %147 = call @Unknown117(%65#2) : (memref<128xf32>) -> memref<128xf32>
    %148:3 = call @BatchNormGradOp118(%17, %arg35, %65#1, %147, %146) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %149 = call @ConvBackwardDataOp119(%148#0, %39) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %150 = call @ConvBackwardFilterOp120(%64#0, %148#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %151 = call @Unknown121(%64#1, %149) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %152 = call @Unknown122(%62#2) : (memref<128xf32>) -> memref<128xf32>
    %153:3 = call @BatchNormGradOp123(%19, %arg30, %62#1, %152, %151) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %154 = call @ConvBackwardDataOp124(%153#0, %38) : (memref<32x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %155 = call @ConvBackwardFilterOp125(%61#0, %153#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %156 = call @Unknown126(%63#2) : (memref<128xf32>) -> memref<128xf32>
    %157:3 = call @BatchNormGradOp127(%18, %arg39, %63#1, %156, %146) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %158 = call @ConvBackwardDataOp128(%157#0, %40) : (memref<32x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16>
    %159 = call @ConvBackwardFilterOp129(%61#0, %157#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %160 = call @Unknown130(%158, %154, %61#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %161 = call @Unknown131(%60#2) : (memref<64xf32>) -> memref<64xf32>
    %162:3 = call @BatchNormGradOp132(%20, %arg25, %60#1, %161, %160) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %163 = call @ConvBackwardDataOp133(%162#0, %37) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %164 = call @ConvBackwardFilterOp134(%59#0, %162#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %165 = call @Unknown135(%59#1, %163) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %166 = call @Unknown136(%58#2) : (memref<64xf32>) -> memref<64xf32>
    %167:3 = call @BatchNormGradOp137(%21, %arg20, %58#1, %166, %165) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %168 = call @ConvBackwardDataOp138(%167#0, %36) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %169 = call @ConvBackwardFilterOp139(%57#0, %167#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %170 = call @Unknown140(%160, %168, %57#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %171 = call @Unknown141(%56#2) : (memref<64xf32>) -> memref<64xf32>
    %172:3 = call @BatchNormGradOp142(%22, %arg15, %56#1, %171, %170) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %173 = call @ConvBackwardDataOp143(%172#0, %35) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %174 = call @ConvBackwardFilterOp144(%55#0, %172#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %175 = call @Unknown145(%55#1, %173) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %176 = call @Unknown146(%54#2) : (memref<64xf32>) -> memref<64xf32>
    %177:3 = call @BatchNormGradOp147(%23, %arg10, %54#1, %176, %175) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %178 = call @ConvBackwardDataOp148(%177#0, %34) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %179 = call @ConvBackwardFilterOp149(%24, %177#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %180 = call @Unknown150(%170, %178) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    "lmhlo.select_and_scatter"(%53#0, %180, %28, %4) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%252) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %252 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%252) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<32x64x56x56xf16>, memref<f16>, memref<32x64x112x112xf16>) -> ()
    %181 = call @Unknown151(%53#1, %4) : (memref<32x64x112x112xi1>, memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16>
    %182 = call @Unknown152(%32#2) : (memref<64xf32>) -> memref<64xf32>
    %183:3 = call @BatchNormGradOp153(%26, %arg5, %32#1, %182, %181) : (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %184 = call @ConvBackwardFilterOp154(%30, %183#0) : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %185 = call @Unknown155(%184) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %186 = call @Unknown156(%179) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %187 = call @Unknown157(%174) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %188 = call @Unknown158(%169) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %189 = call @Unknown159(%164) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %190 = call @Unknown160(%155) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %191 = call @Unknown161(%150) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %192 = call @Unknown162(%159) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %193 = call @Unknown163(%145) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %194 = call @Unknown164(%140) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %195 = call @Unknown165(%131) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %196 = call @Unknown166(%126) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %197 = call @Unknown167(%135) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %198 = call @Unknown168(%121) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %199 = call @Unknown169(%116) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %200 = call @Unknown170(%107) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %201 = call @Unknown171(%102) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %202 = call @Unknown172(%111) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %203 = call @Unknown173(%97) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %204 = call @Unknown174(%92) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    "lmhlo.reduce"(%88#0, %28, %3) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<32x512x7x7xf16>, memref<f16>, memref<32x512xf16>) -> ()
    %205 = call @Unknown175(%3) : (memref<32x512xf16>) -> memref<32x512xf16>
    %206 = call @MatmulOp176(%205, %arg102) : (memref<32x512xf16>, memref<32x1000xf16>) -> memref<1000x512xf16>
    %207 = call @Unknown177(%206) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %208 = call @Unknown178(%arg102) : (memref<32x1000xf16>) -> memref<32x1000xf32>
    "lmhlo.reduce"(%208, %29, %2) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<32x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %209 = call @Unknown179(%2) : (memref<1000xf32>) -> memref<1000xf32>
    %210 = call @Unknown180(%32#1, %arg3) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %211 = call @Unknown181(%32#2, %arg2) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %212 = call @Unknown182(%54#1, %arg8) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %213 = call @Unknown183(%54#2, %arg7) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %214 = call @Unknown184(%56#1, %arg13) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %215 = call @Unknown185(%56#2, %arg12) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %216 = call @Unknown186(%58#1, %arg18) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %217 = call @Unknown187(%58#2, %arg17) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %218 = call @Unknown188(%60#1, %arg23) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %219 = call @Unknown189(%60#2, %arg22) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %220 = call @Unknown190(%62#1, %arg28) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %221 = call @Unknown191(%62#2, %arg27) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %222 = call @Unknown192(%65#1, %arg33) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %223 = call @Unknown193(%65#2, %arg32) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %224 = call @Unknown194(%63#1, %arg37) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %225 = call @Unknown195(%63#2, %arg36) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %226 = call @Unknown196(%67#1, %arg43) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %227 = call @Unknown197(%67#2, %arg42) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %228 = call @Unknown198(%69#1, %arg48) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %229 = call @Unknown199(%69#2, %arg47) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %230 = call @Unknown200(%71#1, %arg53) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %231 = call @Unknown201(%71#2, %arg52) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %232 = call @Unknown202(%74#1, %arg58) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %233 = call @Unknown203(%74#2, %arg57) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %234 = call @Unknown204(%72#1, %arg62) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %235 = call @Unknown205(%72#2, %arg61) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %236 = call @Unknown206(%76#1, %arg68) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %237 = call @Unknown207(%76#2, %arg67) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %238 = call @Unknown208(%78#1, %arg73) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %239 = call @Unknown209(%78#2, %arg72) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %240 = call @Unknown210(%80#1, %arg78) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %241 = call @Unknown211(%80#2, %arg77) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %242 = call @Unknown212(%83#1, %arg83) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %243 = call @Unknown213(%83#2, %arg82) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %244 = call @Unknown214(%81#1, %arg87) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %245 = call @Unknown215(%81#2, %arg86) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %246 = call @Unknown216(%85#1, %arg93) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %247 = call @Unknown217(%85#2, %arg92) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %248 = call @Unknown218(%87#1, %arg98) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %249 = call @Unknown219(%87#2, %arg97) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %250 = call @Unknown220(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    "lmhlo.dot"(%205, %33, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<1000x512xf16>, memref<32x1000xf16>) -> ()
    %251 = call @Unknown221(%250, %1) : (memref<1000xf16>, memref<32x1000xf16>) -> memref<32x1000xf16>
    return %185, %183#1, %183#2, %186, %177#1, %177#2, %187, %172#1, %172#2, %188, %167#1, %167#2, %189, %162#1, %162#2, %190, %153#1, %153#2, %191, %148#1, %148#2, %192, %157#1, %157#2, %193, %143#1, %143#2, %194, %138#1, %138#2, %195, %129#1, %129#2, %196, %124#1, %124#2, %197, %133#1, %133#2, %198, %119#1, %119#2, %199, %114#1, %114#2, %200, %105#1, %105#2, %201, %100#1, %100#2, %202, %109#1, %109#2, %203, %95#1, %95#2, %204, %90#1, %90#2, %207, %209, %210, %211, %0, %212, %213, %0, %214, %215, %0, %216, %217, %0, %218, %219, %0, %220, %221, %0, %222, %223, %0, %224, %225, %0, %226, %227, %0, %228, %229, %0, %230, %231, %0, %232, %233, %0, %234, %235, %0, %236, %237, %0, %238, %239, %0, %240, %241, %0, %242, %243, %0, %244, %245, %0, %246, %247, %0, %248, %249, %0, %251 : memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<32x1000xf16>
  }
}

