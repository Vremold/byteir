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
  func private @BatchNormTrainingOp0(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x112x112xf32>
    %1 = memref.alloc() : memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    return %2, %1 : memref<64xf32>, memref<64xf32>
  }
  func private @BatchNormTrainingOp1(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    return %2, %1 : memref<64xf32>, memref<64xf32>
  }
  func private @BatchNormTrainingOp2(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    return %2, %1 : memref<64xf32>, memref<64xf32>
  }
  func private @BatchNormTrainingOp3(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    return %2, %1 : memref<64xf32>, memref<64xf32>
  }
  func private @BatchNormTrainingOp4(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    return %2, %1 : memref<64xf32>, memref<64xf32>
  }
  func private @BatchNormTrainingOp5(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %2, %1 : memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp6(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %2, %1 : memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp7(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %2, %1 : memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp8(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %2, %1 : memref<128xf32>, memref<128xf32>
  }
  func private @BatchNormTrainingOp9(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<256xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    return %2, %1 : memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp10(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<256xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    return %2, %1 : memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp11(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<256xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    return %2, %1 : memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp12(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<256xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    return %2, %1 : memref<256xf32>, memref<256xf32>
  }
  func private @BatchNormTrainingOp13(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<512xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    return %2, %1 : memref<512xf32>, memref<512xf32>
  }
  func private @BatchNormTrainingOp14(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<512xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    return %2, %1 : memref<512xf32>, memref<512xf32>
  }
  func private @BatchNormTrainingOp15(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<512xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    return %2, %1 : memref<512xf32>, memref<512xf32>
  }
  func private @BatchNormTrainingOp16(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<512xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    return %2, %1 : memref<512xf32>, memref<512xf32>
  }
  func private @Unknown17(%arg0: memref<1x512xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
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
      %9 = select %8, %7, %cst : f16
      affine.store %9, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown18(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp19(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp20(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp21(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.apply #map2(%arg2)
      %4 = affine.apply #map3(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown23(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp24(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp25(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp26(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown27(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>, %arg2: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown28(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp29(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp30(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x512x512xf16>, memref<1x512x7x7xf16>) -> ()
    return %1 : memref<1x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp31(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown32(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x512x7x7xf16>
    affine.for %arg2 = 0 to 25088 {
      %1 = affine.apply #map0(%arg2)
      %2 = affine.apply #map1(%arg2)
      %3 = affine.apply #map2(%arg2)
      %4 = affine.apply #map3(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x512x7x7xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x512x7x7xf16>
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func private @Unknown33(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp34(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp35(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<3x3x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @BatchNormTrainingOp37(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<512xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    return %2, %1 : memref<512xf32>, memref<512xf32>
  }
  func private @Unknown38(%arg0: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp39(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>, %arg3: memref<512xf32>, %arg4: memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    %1 = memref.alloc() : memref<1x512x7x7xf16>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    %4 = memref.alloc() : memref<1x512x7x7xf32>
    %5 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %1, %3, %2 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp40(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x512x7x7xf16>, memref<1x1x256x512xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp41(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown42(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown43(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp44(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp45(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp46(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown47(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map6(%arg2)
      %4 = affine.apply #map7(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown48(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp49(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp50(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp51(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown52(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>, %arg2: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown53(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp54(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp55(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x256x256xf16>, memref<1x256x14x14xf16>) -> ()
    return %1 : memref<1x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp56(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown57(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x256x14x14xf16>
    affine.for %arg2 = 0 to 50176 {
      %1 = affine.apply #map4(%arg2)
      %2 = affine.apply #map5(%arg2)
      %3 = affine.apply #map6(%arg2)
      %4 = affine.apply #map7(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x256x14x14xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x256x14x14xf16>
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func private @Unknown58(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp59(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp60(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<3x3x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp61(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @BatchNormTrainingOp62(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<256xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    return %2, %1 : memref<256xf32>, memref<256xf32>
  }
  func private @Unknown63(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp64(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    %1 = memref.alloc() : memref<1x256x14x14xf16>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    %4 = memref.alloc() : memref<1x256x14x14xf32>
    %5 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %1, %3, %2 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp65(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x256x14x14xf16>, memref<1x1x128x256xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp66(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown67(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown68(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp69(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp70(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp71(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown72(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map8(%arg2)
      %2 = affine.apply #map9(%arg2)
      %3 = affine.apply #map10(%arg2)
      %4 = affine.apply #map11(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown73(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp74(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp75(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp76(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown77(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>, %arg2: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown78(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp79(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x128x128xf16>, memref<1x128x28x28xf16>) -> ()
    return %1 : memref<1x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown82(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x128x28x28xf16>
    affine.for %arg2 = 0 to 100352 {
      %1 = affine.apply #map8(%arg2)
      %2 = affine.apply #map9(%arg2)
      %3 = affine.apply #map10(%arg2)
      %4 = affine.apply #map11(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x128x28x28xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x128x28x28xf16>
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func private @Unknown83(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp84(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp85(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<3x3x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp86(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @BatchNormTrainingOp87(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<128xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %3, %2, %1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %2, %1 : memref<128xf32>, memref<128xf32>
  }
  func private @Unknown88(%arg0: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp89(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    %1 = memref.alloc() : memref<1x128x28x28xf16>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    %4 = memref.alloc() : memref<1x128x28x28xf32>
    %5 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %1, %3, %2 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp90(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x128x28x28xf16>, memref<1x1x64x128xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp91(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown92(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown93(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp94(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown97(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map12(%arg2)
      %2 = affine.apply #map13(%arg2)
      %3 = affine.apply #map14(%arg2)
      %4 = affine.apply #map15(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown98(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp99(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp100(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp101(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown102(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>, %arg2: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
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
      %10 = select %9, %8, %cst : f16
      affine.store %10, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown103(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp104(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp105(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp106(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown107(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x56x56xf16>
    affine.for %arg2 = 0 to 200704 {
      %1 = affine.apply #map12(%arg2)
      %2 = affine.apply #map13(%arg2)
      %3 = affine.apply #map14(%arg2)
      %4 = affine.apply #map15(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x56x56xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x56x56xf16>
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func private @Unknown108(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp109(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x56x56xf32>
    %5 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %1, %3, %2 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<1x64x56x56xf16>
    %2 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.reverse"(%0, %2) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    lmhlo.convolution(%arg0, %2, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<3x3x64x64xf16>, memref<1x64x56x56xf16>) -> ()
    return %1 : memref<1x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<2xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown112(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1x64x56x56xf16>
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
  func private @Unknown113(%arg0: memref<1x64x112x112xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<1x64x112x112xf16>
    affine.for %arg2 = 0 to 802816 {
      %1 = affine.apply #map16(%arg2)
      %2 = affine.apply #map17(%arg2)
      %3 = affine.apply #map18(%arg2)
      %4 = affine.apply #map19(%arg2)
      %5 = affine.load %arg0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
      %6 = affine.load %arg1[%4, %3, %2, %1] : memref<1x64x112x112xf16>
      %7 = arith.cmpf ogt, %5, %cst : f16
      %8 = select %7, %6, %cst : f16
      affine.store %8, %0[%4, %3, %2, %1] : memref<1x64x112x112xf16>
    }
    return %0 : memref<1x64x112x112xf16>
  }
  func private @Unknown114(%arg0: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
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
  func private @BatchNormGradOp115(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<1x64x112x112xf32>
    %1 = memref.alloc() : memref<1x64x112x112xf16>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    %4 = memref.alloc() : memref<1x64x112x112xf32>
    %5 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.convert"(%arg4, %5) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    "lmhlo.batch_norm_grad"(%0, %arg1, %arg2, %arg3, %5, %4, %3, %2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    "lmhlo.convert"(%4, %1) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %1, %3, %2 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp116(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<2xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown117(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x3x7x7xf32>
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
  func private @Unknown118(%arg0: memref<1x1000xf16>) -> memref<1x1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1x1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.apply #map22(%arg1)
      %2 = affine.apply #map23(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1x1000xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1x1000xf32>
    }
    return %0 : memref<1x1000xf32>
  }
  func private @Unknown119(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf32>
    affine.for %arg1 = 0 to 1000 {
      %1 = affine.load %arg0[%arg1] : memref<1000xf32>
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.extf %2 : f16 to f32
      affine.store %3, %0[%arg1] : memref<1000xf32>
    }
    return %0 : memref<1000xf32>
  }
  func private @Unknown120(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf32>
    affine.for %arg1 = 0 to 512000 {
      %1 = affine.apply #map24(%arg1)
      %2 = affine.apply #map25(%arg1)
      %3 = affine.load %arg0[%2, %1] : memref<1000x512xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1] : memref<1000x512xf32>
    }
    return %0 : memref<1000x512xf32>
  }
  func private @Unknown121(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
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
  func private @Unknown122(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
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
  func private @Unknown123(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
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
  func private @Unknown124(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
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
  func private @Unknown125(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf32>
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
  func private @Unknown126(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
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
  func private @Unknown127(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<128x64x1x1xf32>
    affine.for %arg1 = 0 to 8192 {
      %1 = affine.apply #map32(%arg1)
      %2 = affine.apply #map33(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<128x64x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<128x64x1x1xf32>
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func private @Unknown128(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
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
  func private @Unknown129(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
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
  func private @Unknown130(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf32>
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
  func private @Unknown131(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
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
  func private @Unknown132(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<256x128x1x1xf32>
    affine.for %arg1 = 0 to 32768 {
      %1 = affine.apply #map36(%arg1)
      %2 = affine.apply #map37(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<256x128x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<256x128x1x1xf32>
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func private @Unknown133(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
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
  func private @Unknown134(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
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
  func private @Unknown135(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf32>
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
  func private @Unknown136(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
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
  func private @Unknown137(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<512x256x1x1xf32>
    affine.for %arg1 = 0 to 131072 {
      %1 = affine.apply #map40(%arg1)
      %2 = affine.apply #map41(%arg1)
      %3 = affine.load %arg0[%2, %1, %c0, %c0] : memref<512x256x1x1xf16>
      %4 = arith.extf %3 : f16 to f32
      affine.store %4, %0[%2, %1, %c0, %c0] : memref<512x256x1x1xf32>
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func private @Unknown138(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
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
  func private @Unknown139(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
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
    %6:2 = call @BatchNormTrainingOp0(%arg82, %arg1, %arg0) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>)
    %7:2 = call @BatchNormTrainingOp1(%arg86, %arg3, %arg2) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>)
    %8:2 = call @BatchNormTrainingOp2(%arg89, %arg5, %arg4) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>)
    %9:2 = call @BatchNormTrainingOp3(%arg92, %arg7, %arg6) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>)
    %10:2 = call @BatchNormTrainingOp4(%arg95, %arg9, %arg8) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<64xf32>, memref<64xf32>)
    %11:2 = call @BatchNormTrainingOp5(%arg98, %arg11, %arg10) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<128xf32>, memref<128xf32>)
    %12:2 = call @BatchNormTrainingOp6(%arg101, %arg13, %arg12) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<128xf32>, memref<128xf32>)
    %13:2 = call @BatchNormTrainingOp7(%arg106, %arg17, %arg16) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<128xf32>, memref<128xf32>)
    %14:2 = call @BatchNormTrainingOp8(%arg109, %arg19, %arg18) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<128xf32>, memref<128xf32>)
    %15:2 = call @BatchNormTrainingOp9(%arg112, %arg21, %arg20) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<256xf32>, memref<256xf32>)
    %16:2 = call @BatchNormTrainingOp10(%arg115, %arg23, %arg22) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<256xf32>, memref<256xf32>)
    %17:2 = call @BatchNormTrainingOp11(%arg120, %arg27, %arg26) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<256xf32>, memref<256xf32>)
    %18:2 = call @BatchNormTrainingOp12(%arg123, %arg29, %arg28) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<256xf32>, memref<256xf32>)
    %19:2 = call @BatchNormTrainingOp13(%arg126, %arg31, %arg30) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<512xf32>, memref<512xf32>)
    %20:2 = call @BatchNormTrainingOp14(%arg129, %arg33, %arg32) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<512xf32>, memref<512xf32>)
    %21:2 = call @BatchNormTrainingOp15(%arg134, %arg37, %arg36) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<512xf32>, memref<512xf32>)
    %22:2 = call @BatchNormTrainingOp16(%arg137, %arg39, %arg38) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<512xf32>, memref<512xf32>)
    "lmhlo.dot"(%arg141, %arg140, %4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x1000xf16>, memref<512x1000xf16>, memref<1x512xf16>) -> ()
    %23 = call @Unknown17(%4, %arg138) : (memref<1x512xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %24 = call @Unknown18(%22#1) : (memref<512xf32>) -> memref<512xf32>
    %25:3 = call @BatchNormGradOp19(%arg137, %arg39, %22#0, %24, %23) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %26 = call @ConvBackwardDataOp20(%25#0, %arg136) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %27 = call @ConvBackwardFilterOp21(%arg135, %25#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %28 = call @Unknown22(%arg135, %26) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %29 = call @Unknown23(%21#1) : (memref<512xf32>) -> memref<512xf32>
    %30:3 = call @BatchNormGradOp24(%arg134, %arg37, %21#0, %29, %28) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %31 = call @ConvBackwardDataOp25(%30#0, %arg133) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %32 = call @ConvBackwardFilterOp26(%arg132, %30#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %33 = call @Unknown27(%23, %31, %arg132) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %34 = call @Unknown28(%20#1) : (memref<512xf32>) -> memref<512xf32>
    %35:3 = call @BatchNormGradOp29(%arg129, %arg33, %20#0, %34, %33) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %36 = call @ConvBackwardDataOp30(%35#0, %arg128) : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<1x512x7x7xf16>
    %37 = call @ConvBackwardFilterOp31(%arg127, %35#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %38 = call @Unknown32(%arg127, %36) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %39 = call @Unknown33(%19#1) : (memref<512xf32>) -> memref<512xf32>
    %40:3 = call @BatchNormGradOp34(%arg126, %arg31, %19#0, %39, %38) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %41 = call @ConvBackwardDataOp35(%40#0, %arg125) : (memref<1x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %42 = call @ConvBackwardFilterOp36(%arg124, %40#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %43:2 = call @BatchNormTrainingOp37(%arg131, %arg35, %arg34) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<512xf32>, memref<512xf32>)
    %44 = call @Unknown38(%43#1) : (memref<512xf32>) -> memref<512xf32>
    %45:3 = call @BatchNormGradOp39(%arg131, %arg35, %43#0, %44, %33) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf16>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %46 = call @ConvBackwardDataOp40(%45#0, %arg130) : (memref<1x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<1x256x14x14xf16>
    %47 = call @ConvBackwardFilterOp41(%arg124, %45#0) : (memref<1x256x14x14xf16>, memref<1x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %48 = call @Unknown42(%46, %41, %arg124) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %49 = call @Unknown43(%18#1) : (memref<256xf32>) -> memref<256xf32>
    %50:3 = call @BatchNormGradOp44(%arg123, %arg29, %18#0, %49, %48) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %51 = call @ConvBackwardDataOp45(%50#0, %arg122) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %52 = call @ConvBackwardFilterOp46(%arg121, %50#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %53 = call @Unknown47(%arg121, %51) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %54 = call @Unknown48(%17#1) : (memref<256xf32>) -> memref<256xf32>
    %55:3 = call @BatchNormGradOp49(%arg120, %arg27, %17#0, %54, %53) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %56 = call @ConvBackwardDataOp50(%55#0, %arg119) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %57 = call @ConvBackwardFilterOp51(%arg118, %55#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %58 = call @Unknown52(%48, %56, %arg118) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %59 = call @Unknown53(%16#1) : (memref<256xf32>) -> memref<256xf32>
    %60:3 = call @BatchNormGradOp54(%arg115, %arg23, %16#0, %59, %58) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %61 = call @ConvBackwardDataOp55(%60#0, %arg114) : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<1x256x14x14xf16>
    %62 = call @ConvBackwardFilterOp56(%arg113, %60#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %63 = call @Unknown57(%arg113, %61) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %64 = call @Unknown58(%15#1) : (memref<256xf32>) -> memref<256xf32>
    %65:3 = call @BatchNormGradOp59(%arg112, %arg21, %15#0, %64, %63) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %66 = call @ConvBackwardDataOp60(%65#0, %arg111) : (memref<1x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %67 = call @ConvBackwardFilterOp61(%arg110, %65#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %68:2 = call @BatchNormTrainingOp62(%arg117, %arg25, %arg24) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<256xf32>, memref<256xf32>)
    %69 = call @Unknown63(%68#1) : (memref<256xf32>) -> memref<256xf32>
    %70:3 = call @BatchNormGradOp64(%arg117, %arg25, %68#0, %69, %58) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf16>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %71 = call @ConvBackwardDataOp65(%70#0, %arg116) : (memref<1x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<1x128x28x28xf16>
    %72 = call @ConvBackwardFilterOp66(%arg110, %70#0) : (memref<1x128x28x28xf16>, memref<1x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %73 = call @Unknown67(%71, %66, %arg110) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %74 = call @Unknown68(%14#1) : (memref<128xf32>) -> memref<128xf32>
    %75:3 = call @BatchNormGradOp69(%arg109, %arg19, %14#0, %74, %73) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %76 = call @ConvBackwardDataOp70(%75#0, %arg108) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %77 = call @ConvBackwardFilterOp71(%arg107, %75#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %78 = call @Unknown72(%arg107, %76) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %79 = call @Unknown73(%13#1) : (memref<128xf32>) -> memref<128xf32>
    %80:3 = call @BatchNormGradOp74(%arg106, %arg17, %13#0, %79, %78) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %81 = call @ConvBackwardDataOp75(%80#0, %arg105) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %82 = call @ConvBackwardFilterOp76(%arg104, %80#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %83 = call @Unknown77(%73, %81, %arg104) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %84 = call @Unknown78(%12#1) : (memref<128xf32>) -> memref<128xf32>
    %85:3 = call @BatchNormGradOp79(%arg101, %arg13, %12#0, %84, %83) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %86 = call @ConvBackwardDataOp80(%85#0, %arg100) : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<1x128x28x28xf16>
    %87 = call @ConvBackwardFilterOp81(%arg99, %85#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %88 = call @Unknown82(%arg99, %86) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %89 = call @Unknown83(%11#1) : (memref<128xf32>) -> memref<128xf32>
    %90:3 = call @BatchNormGradOp84(%arg98, %arg11, %11#0, %89, %88) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %91 = call @ConvBackwardDataOp85(%90#0, %arg97) : (memref<1x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %92 = call @ConvBackwardFilterOp86(%arg96, %90#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %93:2 = call @BatchNormTrainingOp87(%arg103, %arg15, %arg14) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<128xf32>, memref<128xf32>)
    %94 = call @Unknown88(%93#1) : (memref<128xf32>) -> memref<128xf32>
    %95:3 = call @BatchNormGradOp89(%arg103, %arg15, %93#0, %94, %83) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf16>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %96 = call @ConvBackwardDataOp90(%95#0, %arg102) : (memref<1x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<1x64x56x56xf16>
    %97 = call @ConvBackwardFilterOp91(%arg96, %95#0) : (memref<1x64x56x56xf16>, memref<1x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %98 = call @Unknown92(%96, %91, %arg96) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %99 = call @Unknown93(%10#1) : (memref<64xf32>) -> memref<64xf32>
    %100:3 = call @BatchNormGradOp94(%arg95, %arg9, %10#0, %99, %98) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %101 = call @ConvBackwardDataOp95(%100#0, %arg94) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %102 = call @ConvBackwardFilterOp96(%arg93, %100#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %103 = call @Unknown97(%arg93, %101) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %104 = call @Unknown98(%9#1) : (memref<64xf32>) -> memref<64xf32>
    %105:3 = call @BatchNormGradOp99(%arg92, %arg7, %9#0, %104, %103) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %106 = call @ConvBackwardDataOp100(%105#0, %arg91) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %107 = call @ConvBackwardFilterOp101(%arg90, %105#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %108 = call @Unknown102(%98, %106, %arg90) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %109 = call @Unknown103(%8#1) : (memref<64xf32>) -> memref<64xf32>
    %110:3 = call @BatchNormGradOp104(%arg89, %arg5, %8#0, %109, %108) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %111 = call @ConvBackwardDataOp105(%110#0, %arg88) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %112 = call @ConvBackwardFilterOp106(%arg87, %110#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %113 = call @Unknown107(%arg87, %111) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %114 = call @Unknown108(%7#1) : (memref<64xf32>) -> memref<64xf32>
    %115:3 = call @BatchNormGradOp109(%arg86, %arg3, %7#0, %114, %113) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf16>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %116 = call @ConvBackwardDataOp110(%115#0, %arg85) : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<1x64x56x56xf16>
    %117 = call @ConvBackwardFilterOp111(%arg84, %115#0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %118 = call @Unknown112(%108, %116) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    "lmhlo.select_and_scatter"(%arg83, %118, %5, %3) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):  // no predecessors
      %146 = "mhlo.compare"(%arg142, %arg143) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%146) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):  // no predecessors
      %146 = mhlo.add %arg142, %arg143 : tensor<f16>
      "mhlo.return"(%146) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<f16>, memref<1x64x112x112xf16>) -> ()
    %119 = call @Unknown113(%arg83, %3) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %120 = call @Unknown114(%6#1) : (memref<64xf32>) -> memref<64xf32>
    %121:3 = call @BatchNormGradOp115(%arg82, %arg1, %6#0, %120, %119) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf16>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %122 = call @ConvBackwardFilterOp116(%arg81, %121#0) : (memref<1x3x224x224xf16>, memref<1x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %123 = call @Unknown117(%122) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %124 = call @Unknown118(%arg141) : (memref<1x1000xf16>) -> memref<1x1000xf32>
    "lmhlo.reduce"(%124, %0, %2) ({
    ^bb0(%arg142: memref<f32>, %arg143: memref<f32>, %arg144: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg142, %arg143, %arg144) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<1x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %125 = call @Unknown119(%2) : (memref<1000xf32>) -> memref<1000xf32>
    "lmhlo.dot"(%arg141, %arg139, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<1x1000xf16>, memref<1x512xf16>, memref<1000x512xf16>) -> ()
    %126 = call @Unknown120(%1) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %127 = call @Unknown121(%117) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %128 = call @Unknown122(%112) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %129 = call @Unknown123(%107) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %130 = call @Unknown124(%102) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %131 = call @Unknown125(%92) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %132 = call @Unknown126(%87) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %133 = call @Unknown127(%97) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %134 = call @Unknown128(%82) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %135 = call @Unknown129(%77) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %136 = call @Unknown130(%67) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %137 = call @Unknown131(%62) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %138 = call @Unknown132(%72) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %139 = call @Unknown133(%57) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %140 = call @Unknown134(%52) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %141 = call @Unknown135(%42) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %142 = call @Unknown136(%37) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %143 = call @Unknown137(%47) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %144 = call @Unknown138(%32) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %145 = call @Unknown139(%27) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    return %121#2, %121#1, %123, %125, %126, %115#2, %115#1, %110#2, %110#1, %127, %128, %105#2, %105#1, %100#2, %100#1, %129, %130, %90#2, %90#1, %85#2, %85#1, %131, %132, %133, %95#2, %95#1, %80#2, %80#1, %75#2, %75#1, %134, %135, %65#2, %65#1, %60#2, %60#1, %136, %137, %138, %70#2, %70#1, %55#2, %55#1, %50#2, %50#1, %139, %140, %40#2, %40#1, %35#2, %35#1, %141, %142, %143, %45#2, %45#1, %30#2, %30#1, %25#2, %25#1, %144, %145 : memref<64xf32>, memref<64xf32>, memref<64x3x7x7xf32>, memref<1000xf32>, memref<1000x512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x3x3xf32>, memref<128x128x3x3xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x3x3xf32>, memref<256x256x3x3xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x3x3xf32>, memref<512x512x3x3xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512x512x3x3xf32>
  }
}

