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
  func private @BatchNormGradOp59(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    %6 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp60(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp61(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown62(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp63(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    %6 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp64(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp65(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown66(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>, %arg2: memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp67(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    %6 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp68(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x512x512xf16>, memref<32x512x7x7xf16>) -> ()
    return %1 : memref<32x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp69(%arg0: memref<32x512x7x7xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown70(%arg0: memref<32x512x7x7xi1>, %arg1: memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp71(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    %6 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp72(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<3x3x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp73(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @BatchNormGradOp74(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    %1 = memref.alloc() : memref<32x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<32x512x7x7xf32>
    %5 = memref.alloc() : memref<32x512x7x7xf32>
    %6 = memref.alloc() : memref<32x512x7x7xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<32x512x7x7xf32>, memref<32x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x512x7x7xf32>, memref<32x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp75(%arg0: memref<32x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512x7x7xf16>, memref<1x1x256x512xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp76(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown77(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp78(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    %6 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp79(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp80(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown81(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp82(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    %6 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp83(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp84(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown85(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>, %arg2: memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp86(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    %6 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp87(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x256x256xf16>, memref<32x256x14x14xf16>) -> ()
    return %1 : memref<32x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp88(%arg0: memref<32x256x14x14xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown89(%arg0: memref<32x256x14x14xi1>, %arg1: memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp90(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    %6 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp91(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<3x3x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp92(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @BatchNormGradOp93(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    %1 = memref.alloc() : memref<32x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<32x256x14x14xf32>
    %5 = memref.alloc() : memref<32x256x14x14xf32>
    %6 = memref.alloc() : memref<32x256x14x14xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<32x256x14x14xf32>, memref<32x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x256x14x14xf32>, memref<32x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp94(%arg0: memref<32x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x256x14x14xf16>, memref<1x1x128x256xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp95(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown96(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp97(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    %6 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp98(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp99(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown100(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp101(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    %6 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp102(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp103(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown104(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>, %arg2: memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp105(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    %6 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp106(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x128x128xf16>, memref<32x128x28x28xf16>) -> ()
    return %1 : memref<32x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp107(%arg0: memref<32x128x28x28xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown108(%arg0: memref<32x128x28x28xi1>, %arg1: memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp109(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    %6 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<3x3x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @BatchNormGradOp112(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    %1 = memref.alloc() : memref<32x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<32x128x28x28xf32>
    %5 = memref.alloc() : memref<32x128x28x28xf32>
    %6 = memref.alloc() : memref<32x128x28x28xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<32x128x28x28xf32>, memref<32x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x128x28x28xf32>, memref<32x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp113(%arg0: memref<32x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x128x28x28xf16>, memref<1x1x64x128xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp114(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown115(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp116(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    %6 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp117(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp118(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown119(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp120(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    %6 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp121(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp122(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown123(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>, %arg2: memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp124(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    %6 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp125(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp126(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown127(%arg0: memref<32x64x56x56xi1>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp128(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x56x56xf32>
    %5 = memref.alloc() : memref<32x64x56x56xf32>
    %6 = memref.alloc() : memref<32x64x56x56xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x56x56xf32>, memref<32x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x56x56xf32>, memref<32x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp129(%arg0: memref<32x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<32x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<3x3x64x64xf16>, memref<32x64x56x56xf16>) -> ()
    return %1 : memref<32x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp130(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown131(%arg0: memref<32x64x56x56xf16>, %arg1: memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown132(%arg0: memref<32x64x112x112xi1>, %arg1: memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp133(%arg0: memref<32x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    %1 = memref.alloc() : memref<32x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<32x64x112x112xf32>
    %5 = memref.alloc() : memref<32x64x112x112xf32>
    %6 = memref.alloc() : memref<32x64x112x112xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    "lmhlo.convert"(%arg0, %6) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg2, %5) : (memref<32x64x112x112xf16>, memref<32x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%6, %arg1, %0, %0, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<32x64x112x112xf32>, memref<32x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<32x64x112x112xf32>, memref<32x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp134(%arg0: memref<32x3x224x224xf16>, %arg1: memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown135(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown136(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown137(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown138(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown139(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown140(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown141(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown142(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown143(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown144(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown145(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown146(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown147(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown148(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown149(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown150(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown151(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown152(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown153(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown154(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown155(%arg0: memref<32x512xf16>) -> memref<32x512xf16> attributes {__byteir_elementwise_fusion__} {
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
  func private @MatmulOp156(%arg0: memref<32x512xf16>, %arg1: memref<32x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<512x1000xf16>
    %1 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<32x1000xf16>, memref<512x1000xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %1 : memref<1000x512xf16>
  }
  func private @Unknown157(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown158(%arg0: memref<32x1000xf16>) -> memref<32x1000xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown159(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      affine.store %3, %0[%arg1] : memref<1000xf32>
    }
    return %0 : memref<1000xf32>
  }
  func private @Unknown160(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown161(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown162(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown163(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown164(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown165(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown166(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown167(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown168(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown169(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown170(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown171(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown172(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown173(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown174(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown175(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown176(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown177(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown178(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown179(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown180(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown181(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown182(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown183(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown184(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown185(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown186(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown187(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown188(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown189(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown190(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown191(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown192(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown193(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown194(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown195(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown196(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown197(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown198(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown199(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @Unknown200(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf16>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      affine.store %2, %0[%arg1] : memref<1000xf16>
    }
    return %0 : memref<1000xf16>
  }
  func private @Unknown201(%arg0: memref<1000xf16>, %arg1: memref<32x1000xf16>) -> memref<32x1000xf16> attributes {__byteir_elementwise_fusion__} {
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
      %232 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %232) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%232, %arg106) : (memref<f16>, memref<f16>) -> ()
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
    %89:3 = call @BatchNormGradOp59(%5, %arg100, %88#1) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %90 = call @ConvBackwardDataOp60(%89#0, %52) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %91 = call @ConvBackwardFilterOp61(%86#0, %89#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %92 = call @Unknown62(%86#1, %90) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %93:3 = call @BatchNormGradOp63(%6, %arg95, %92) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %94 = call @ConvBackwardDataOp64(%93#0, %51) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %95 = call @ConvBackwardFilterOp65(%84#0, %93#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %96 = call @Unknown66(%88#1, %94, %84#1) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>, memref<32x512x7x7xi1>) -> memref<32x512x7x7xf16>
    %97:3 = call @BatchNormGradOp67(%7, %arg85, %96) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %98 = call @ConvBackwardDataOp68(%97#0, %49) : (memref<32x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<32x512x7x7xf16>
    %99 = call @ConvBackwardFilterOp69(%82#0, %97#0) : (memref<32x512x7x7xf16>, memref<32x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %100 = call @Unknown70(%82#1, %98) : (memref<32x512x7x7xi1>, memref<32x512x7x7xf16>) -> memref<32x512x7x7xf16>
    %101:3 = call @BatchNormGradOp71(%9, %arg80, %100) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %102 = call @ConvBackwardDataOp72(%101#0, %48) : (memref<32x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %103 = call @ConvBackwardFilterOp73(%79#0, %101#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %104:3 = call @BatchNormGradOp74(%8, %arg89, %96) : (memref<32x512x7x7xf16>, memref<512xf32>, memref<32x512x7x7xf16>) -> (memref<32x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %105 = call @ConvBackwardDataOp75(%104#0, %50) : (memref<32x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<32x256x14x14xf16>
    %106 = call @ConvBackwardFilterOp76(%79#0, %104#0) : (memref<32x256x14x14xf16>, memref<32x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %107 = call @Unknown77(%105, %102, %79#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %108:3 = call @BatchNormGradOp78(%10, %arg75, %107) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %109 = call @ConvBackwardDataOp79(%108#0, %47) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %110 = call @ConvBackwardFilterOp80(%77#0, %108#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %111 = call @Unknown81(%77#1, %109) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %112:3 = call @BatchNormGradOp82(%11, %arg70, %111) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %113 = call @ConvBackwardDataOp83(%112#0, %46) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %114 = call @ConvBackwardFilterOp84(%75#0, %112#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %115 = call @Unknown85(%107, %113, %75#1) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>, memref<32x256x14x14xi1>) -> memref<32x256x14x14xf16>
    %116:3 = call @BatchNormGradOp86(%12, %arg60, %115) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %117 = call @ConvBackwardDataOp87(%116#0, %44) : (memref<32x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<32x256x14x14xf16>
    %118 = call @ConvBackwardFilterOp88(%73#0, %116#0) : (memref<32x256x14x14xf16>, memref<32x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %119 = call @Unknown89(%73#1, %117) : (memref<32x256x14x14xi1>, memref<32x256x14x14xf16>) -> memref<32x256x14x14xf16>
    %120:3 = call @BatchNormGradOp90(%14, %arg55, %119) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %121 = call @ConvBackwardDataOp91(%120#0, %43) : (memref<32x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %122 = call @ConvBackwardFilterOp92(%70#0, %120#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %123:3 = call @BatchNormGradOp93(%13, %arg64, %115) : (memref<32x256x14x14xf16>, memref<256xf32>, memref<32x256x14x14xf16>) -> (memref<32x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %124 = call @ConvBackwardDataOp94(%123#0, %45) : (memref<32x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<32x128x28x28xf16>
    %125 = call @ConvBackwardFilterOp95(%70#0, %123#0) : (memref<32x128x28x28xf16>, memref<32x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %126 = call @Unknown96(%124, %121, %70#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %127:3 = call @BatchNormGradOp97(%15, %arg50, %126) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %128 = call @ConvBackwardDataOp98(%127#0, %42) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %129 = call @ConvBackwardFilterOp99(%68#0, %127#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %130 = call @Unknown100(%68#1, %128) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %131:3 = call @BatchNormGradOp101(%16, %arg45, %130) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %132 = call @ConvBackwardDataOp102(%131#0, %41) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %133 = call @ConvBackwardFilterOp103(%66#0, %131#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %134 = call @Unknown104(%126, %132, %66#1) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>, memref<32x128x28x28xi1>) -> memref<32x128x28x28xf16>
    %135:3 = call @BatchNormGradOp105(%17, %arg35, %134) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %136 = call @ConvBackwardDataOp106(%135#0, %39) : (memref<32x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<32x128x28x28xf16>
    %137 = call @ConvBackwardFilterOp107(%64#0, %135#0) : (memref<32x128x28x28xf16>, memref<32x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %138 = call @Unknown108(%64#1, %136) : (memref<32x128x28x28xi1>, memref<32x128x28x28xf16>) -> memref<32x128x28x28xf16>
    %139:3 = call @BatchNormGradOp109(%19, %arg30, %138) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %140 = call @ConvBackwardDataOp110(%139#0, %38) : (memref<32x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %141 = call @ConvBackwardFilterOp111(%61#0, %139#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %142:3 = call @BatchNormGradOp112(%18, %arg39, %134) : (memref<32x128x28x28xf16>, memref<128xf32>, memref<32x128x28x28xf16>) -> (memref<32x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %143 = call @ConvBackwardDataOp113(%142#0, %40) : (memref<32x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<32x64x56x56xf16>
    %144 = call @ConvBackwardFilterOp114(%61#0, %142#0) : (memref<32x64x56x56xf16>, memref<32x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %145 = call @Unknown115(%143, %140, %61#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %146:3 = call @BatchNormGradOp116(%20, %arg25, %145) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %147 = call @ConvBackwardDataOp117(%146#0, %37) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %148 = call @ConvBackwardFilterOp118(%59#0, %146#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %149 = call @Unknown119(%59#1, %147) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %150:3 = call @BatchNormGradOp120(%21, %arg20, %149) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %151 = call @ConvBackwardDataOp121(%150#0, %36) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %152 = call @ConvBackwardFilterOp122(%57#0, %150#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %153 = call @Unknown123(%145, %151, %57#1) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>, memref<32x64x56x56xi1>) -> memref<32x64x56x56xf16>
    %154:3 = call @BatchNormGradOp124(%22, %arg15, %153) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %155 = call @ConvBackwardDataOp125(%154#0, %35) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %156 = call @ConvBackwardFilterOp126(%55#0, %154#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %157 = call @Unknown127(%55#1, %155) : (memref<32x64x56x56xi1>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    %158:3 = call @BatchNormGradOp128(%23, %arg10, %157) : (memref<32x64x56x56xf16>, memref<64xf32>, memref<32x64x56x56xf16>) -> (memref<32x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %159 = call @ConvBackwardDataOp129(%158#0, %34) : (memref<32x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<32x64x56x56xf16>
    %160 = call @ConvBackwardFilterOp130(%24, %158#0) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %161 = call @Unknown131(%153, %159) : (memref<32x64x56x56xf16>, memref<32x64x56x56xf16>) -> memref<32x64x56x56xf16>
    "lmhlo.select_and_scatter"(%53#0, %161, %28, %4) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %232 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%232) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %232 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%232) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<32x64x56x56xf16>, memref<f16>, memref<32x64x112x112xf16>) -> ()
    %162 = call @Unknown132(%53#1, %4) : (memref<32x64x112x112xi1>, memref<32x64x112x112xf16>) -> memref<32x64x112x112xf16>
    %163:3 = call @BatchNormGradOp133(%26, %arg5, %162) : (memref<32x64x112x112xf16>, memref<64xf32>, memref<32x64x112x112xf16>) -> (memref<32x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %164 = call @ConvBackwardFilterOp134(%30, %163#0) : (memref<32x3x224x224xf16>, memref<32x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %165 = call @Unknown135(%164) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %166 = call @Unknown136(%160) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %167 = call @Unknown137(%156) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %168 = call @Unknown138(%152) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %169 = call @Unknown139(%148) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %170 = call @Unknown140(%141) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %171 = call @Unknown141(%137) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %172 = call @Unknown142(%144) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %173 = call @Unknown143(%133) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %174 = call @Unknown144(%129) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %175 = call @Unknown145(%122) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %176 = call @Unknown146(%118) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %177 = call @Unknown147(%125) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %178 = call @Unknown148(%114) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %179 = call @Unknown149(%110) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %180 = call @Unknown150(%103) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %181 = call @Unknown151(%99) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %182 = call @Unknown152(%106) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %183 = call @Unknown153(%95) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %184 = call @Unknown154(%91) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    "lmhlo.reduce"(%88#0, %28, %3) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<32x512x7x7xf16>, memref<f16>, memref<32x512xf16>) -> ()
    %185 = call @Unknown155(%3) : (memref<32x512xf16>) -> memref<32x512xf16>
    %186 = call @MatmulOp156(%185, %arg102) : (memref<32x512xf16>, memref<32x1000xf16>) -> memref<1000x512xf16>
    %187 = call @Unknown157(%186) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %188 = call @Unknown158(%arg102) : (memref<32x1000xf16>) -> memref<32x1000xf32>
    "lmhlo.reduce"(%188, %29, %2) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<32x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %189 = call @Unknown159(%2) : (memref<1000xf32>) -> memref<1000xf32>
    %190 = call @Unknown160(%32#1, %arg3) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %191 = call @Unknown161(%32#2, %arg2) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %192 = call @Unknown162(%54#1, %arg8) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %193 = call @Unknown163(%54#2, %arg7) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %194 = call @Unknown164(%56#1, %arg13) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %195 = call @Unknown165(%56#2, %arg12) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %196 = call @Unknown166(%58#1, %arg18) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %197 = call @Unknown167(%58#2, %arg17) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %198 = call @Unknown168(%60#1, %arg23) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %199 = call @Unknown169(%60#2, %arg22) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %200 = call @Unknown170(%62#1, %arg28) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %201 = call @Unknown171(%62#2, %arg27) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %202 = call @Unknown172(%65#1, %arg33) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %203 = call @Unknown173(%65#2, %arg32) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %204 = call @Unknown174(%63#1, %arg37) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %205 = call @Unknown175(%63#2, %arg36) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %206 = call @Unknown176(%67#1, %arg43) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %207 = call @Unknown177(%67#2, %arg42) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %208 = call @Unknown178(%69#1, %arg48) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %209 = call @Unknown179(%69#2, %arg47) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %210 = call @Unknown180(%71#1, %arg53) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %211 = call @Unknown181(%71#2, %arg52) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %212 = call @Unknown182(%74#1, %arg58) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %213 = call @Unknown183(%74#2, %arg57) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %214 = call @Unknown184(%72#1, %arg62) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %215 = call @Unknown185(%72#2, %arg61) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %216 = call @Unknown186(%76#1, %arg68) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %217 = call @Unknown187(%76#2, %arg67) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %218 = call @Unknown188(%78#1, %arg73) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %219 = call @Unknown189(%78#2, %arg72) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %220 = call @Unknown190(%80#1, %arg78) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %221 = call @Unknown191(%80#2, %arg77) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %222 = call @Unknown192(%83#1, %arg83) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %223 = call @Unknown193(%83#2, %arg82) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %224 = call @Unknown194(%81#1, %arg87) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %225 = call @Unknown195(%81#2, %arg86) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %226 = call @Unknown196(%85#1, %arg93) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %227 = call @Unknown197(%85#2, %arg92) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %228 = call @Unknown198(%87#1, %arg98) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %229 = call @Unknown199(%87#2, %arg97) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %230 = call @Unknown200(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    "lmhlo.dot"(%185, %33, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<32x512xf16>, memref<1000x512xf16>, memref<32x1000xf16>) -> ()
    %231 = call @Unknown201(%230, %1) : (memref<1000xf16>, memref<32x1000xf16>) -> memref<32x1000xf16>
    return %165, %163#1, %163#2, %166, %158#1, %158#2, %167, %154#1, %154#2, %168, %150#1, %150#2, %169, %146#1, %146#2, %170, %139#1, %139#2, %171, %135#1, %135#2, %172, %142#1, %142#2, %173, %131#1, %131#2, %174, %127#1, %127#2, %175, %120#1, %120#2, %176, %116#1, %116#2, %177, %123#1, %123#2, %178, %112#1, %112#2, %179, %108#1, %108#2, %180, %101#1, %101#2, %181, %97#1, %97#2, %182, %104#1, %104#2, %183, %93#1, %93#2, %184, %89#1, %89#2, %187, %189, %190, %191, %0, %192, %193, %0, %194, %195, %0, %196, %197, %0, %198, %199, %0, %200, %201, %0, %202, %203, %0, %204, %205, %0, %206, %207, %0, %208, %209, %0, %210, %211, %0, %212, %213, %0, %214, %215, %0, %216, %217, %0, %218, %219, %0, %220, %221, %0, %222, %223, %0, %224, %225, %0, %226, %227, %0, %228, %229, %0, %231 : memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<64xf32>, memref<64xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<128xf32>, memref<128xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<256xf32>, memref<256xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<512xf32>, memref<512xf32>, memref<i64>, memref<32x1000xf16>
  }
}

