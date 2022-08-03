// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func.func @main
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func private @Unknown0(%arg0: memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x3x224x224xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x3x224x224xf32>) outs(%0 : memref<1x3x224x224xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x3x224x224xf16>
  }
  func.func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x3x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf32>) outs(%0 : memref<64x3x7x7xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x3x7x7xf16>
  }
  func.func private @BatchNormTrainingOp2(%arg0: memref<1x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x112x112xf16>, memref<1x64x112x112xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x112x112xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<1x64x112x112xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x64x112x112xf32>, memref<1x64x112x112xf16>) -> ()
    return %4, %2, %3 : memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @Unknown3(%arg0: memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x112x112xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x64x112x112xf16>) outs(%0 : memref<1x64x112x112xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x64x112x112xf16>
  }
  func.func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp5(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %4, %2, %3 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @Unknown6(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @Unknown7(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp8(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %4, %2, %3 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @Unknown9(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @Unknown10(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp11(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %4, %2, %3 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @Unknown12(%arg0: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @Unknown13(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp14(%arg0: memref<1x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<1x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x64x56x56xf32>, memref<1x64x56x56xf16>) -> ()
    return %4, %2, %3 : memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func.func private @Unknown15(%arg0: memref<1x64x56x56xf16>, %arg1: memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) outs(%0 : memref<1x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x64x56x56xf16>
  }
  func.func private @Unknown16(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf32>) outs(%0 : memref<128x64x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x64x1x1xf16>
  }
  func.func private @BatchNormTrainingOp17(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %4, %2, %3 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @Unknown18(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf32>) outs(%0 : memref<128x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x64x3x3xf16>
  }
  func.func private @BatchNormTrainingOp19(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %4, %2, %3 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @Unknown20(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @Unknown21(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp22(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %4, %2, %3 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @Unknown23(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @Unknown24(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp25(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %4, %2, %3 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @Unknown26(%arg0: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @Unknown27(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp28(%arg0: memref<1x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<1x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<1x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x128x28x28xf32>, memref<1x128x28x28xf16>) -> ()
    return %4, %2, %3 : memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func.func private @Unknown29(%arg0: memref<1x128x28x28xf16>, %arg1: memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) outs(%0 : memref<1x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x128x28x28xf16>
  }
  func.func private @Unknown30(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf32>) outs(%0 : memref<256x128x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x128x1x1xf16>
  }
  func.func private @BatchNormTrainingOp31(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %4, %2, %3 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @Unknown32(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf32>) outs(%0 : memref<256x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x128x3x3xf16>
  }
  func.func private @BatchNormTrainingOp33(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %4, %2, %3 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @Unknown34(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @Unknown35(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp36(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %4, %2, %3 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @Unknown37(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @Unknown38(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp39(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %4, %2, %3 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @Unknown40(%arg0: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @Unknown41(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp42(%arg0: memref<1x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<1x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<1x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<1x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x256x14x14xf32>, memref<1x256x14x14xf16>) -> ()
    return %4, %2, %3 : memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func.func private @Unknown43(%arg0: memref<1x256x14x14xf16>, %arg1: memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) outs(%0 : memref<1x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x256x14x14xf16>
  }
  func.func private @Unknown44(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf32>) outs(%0 : memref<512x256x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x256x1x1xf16>
  }
  func.func private @BatchNormTrainingOp45(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %4, %2, %3 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @Unknown46(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf32>) outs(%0 : memref<512x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x256x3x3xf16>
  }
  func.func private @BatchNormTrainingOp47(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %4, %2, %3 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @Unknown48(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @Unknown49(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp50(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %4, %2, %3 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @Unknown51(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @Unknown52(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp53(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %4, %2, %3 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @Unknown54(%arg0: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.maxf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @Unknown55(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func.func private @BatchNormTrainingOp56(%arg0: memref<1x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<1x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<1x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<1x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<1x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<1x512x7x7xf32>, memref<1x512x7x7xf16>) -> ()
    return %4, %2, %3 : memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func.func private @Unknown57(%arg0: memref<1x512x7x7xf16>, %arg1: memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) outs(%0 : memref<1x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):
      %1 = arith.addf %arg2, %arg3 : f16
      %2 = arith.maxf %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<1x512x7x7xf16>
  }
  func.func private @Unknown58(%arg0: memref<1x512xf16>) -> memref<1x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x512xf16>) outs(%0 : memref<1x512xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.mulf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1x512xf16>
  }
  func.func private @Unknown59(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1000x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf32>) outs(%0 : memref<1000x512xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1000x512xf16>
  }
  func.func private @Unknown60(%arg0: memref<1000xf32>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.expand_shape %arg0 [[0, 1]] : memref<1000xf32> into memref<1x1000xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<1x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %0 : memref<1x1000xf16>, memref<1x1000xf32>) outs(%1 : memref<1x1000xf16>) {
    ^bb0(%arg2: f16, %arg3: f32, %arg4: f16):
      %2 = arith.truncf %arg3 : f32 to f16
      %3 = arith.addf %arg2, %2 : f16
      linalg.yield %3 : f16
    }
    return %1 : memref<1x1000xf16>
  }
  func.func private @Unknown61(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown62(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown63(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown64(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown65(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown66(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown67(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown68(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown69(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown70(%arg0: memref<64xf32>, %arg1: memref<64xf32>) -> memref<64xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<64xf32>, memref<64xf32>) outs(%0 : memref<64xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<64xf32>
  }
  func.func private @Unknown71(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown72(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown73(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown74(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown75(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown76(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown77(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown78(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown79(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown80(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>) outs(%0 : memref<128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<128xf32>
  }
  func.func private @Unknown81(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown82(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown83(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown84(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown85(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown86(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown87(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown88(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown89(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown90(%arg0: memref<256xf32>, %arg1: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<256xf32>, memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<256xf32>
  }
  func.func private @Unknown91(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown92(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown93(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown94(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown95(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown96(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown97(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown98(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown99(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func private @Unknown100(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> memref<512xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e-01 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<512xf32>, memref<512xf32>) outs(%0 : memref<512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.mulf %arg3, %cst_0 : f32
      %2 = arith.mulf %arg2, %cst : f32
      %3 = arith.addf %2, %1 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<512xf32>
  }
  func.func @main(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<1000xf32>, %arg4: memref<1000x512xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: memref<64xf32>, %arg9: memref<64x64x3x3xf32>, %arg10: memref<64x64x3x3xf32>, %arg11: memref<64xf32>, %arg12: memref<64xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64x64x3x3xf32>, %arg16: memref<64x64x3x3xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<128xf32>, %arg20: memref<128xf32>, %arg21: memref<128x64x3x3xf32>, %arg22: memref<128x128x3x3xf32>, %arg23: memref<128x64x1x1xf32>, %arg24: memref<128xf32>, %arg25: memref<128xf32>, %arg26: memref<128xf32>, %arg27: memref<128xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128x128x3x3xf32>, %arg31: memref<128x128x3x3xf32>, %arg32: memref<256xf32>, %arg33: memref<256xf32>, %arg34: memref<256xf32>, %arg35: memref<256xf32>, %arg36: memref<256x128x3x3xf32>, %arg37: memref<256x256x3x3xf32>, %arg38: memref<256x128x1x1xf32>, %arg39: memref<256xf32>, %arg40: memref<256xf32>, %arg41: memref<256xf32>, %arg42: memref<256xf32>, %arg43: memref<256xf32>, %arg44: memref<256xf32>, %arg45: memref<256x256x3x3xf32>, %arg46: memref<256x256x3x3xf32>, %arg47: memref<512xf32>, %arg48: memref<512xf32>, %arg49: memref<512xf32>, %arg50: memref<512xf32>, %arg51: memref<512x256x3x3xf32>, %arg52: memref<512x512x3x3xf32>, %arg53: memref<512x256x1x1xf32>, %arg54: memref<512xf32>, %arg55: memref<512xf32>, %arg56: memref<512xf32>, %arg57: memref<512xf32>, %arg58: memref<512xf32>, %arg59: memref<512xf32>, %arg60: memref<512x512x3x3xf32>, %arg61: memref<512x512x3x3xf32>, %arg62: memref<i64>, %arg63: memref<64xf32>, %arg64: memref<64xf32>, %arg65: memref<i64>, %arg66: memref<64xf32>, %arg67: memref<64xf32>, %arg68: memref<i64>, %arg69: memref<64xf32>, %arg70: memref<64xf32>, %arg71: memref<i64>, %arg72: memref<64xf32>, %arg73: memref<64xf32>, %arg74: memref<i64>, %arg75: memref<64xf32>, %arg76: memref<64xf32>, %arg77: memref<i64>, %arg78: memref<128xf32>, %arg79: memref<128xf32>, %arg80: memref<i64>, %arg81: memref<128xf32>, %arg82: memref<128xf32>, %arg83: memref<i64>, %arg84: memref<128xf32>, %arg85: memref<128xf32>, %arg86: memref<i64>, %arg87: memref<128xf32>, %arg88: memref<128xf32>, %arg89: memref<i64>, %arg90: memref<128xf32>, %arg91: memref<128xf32>, %arg92: memref<i64>, %arg93: memref<256xf32>, %arg94: memref<256xf32>, %arg95: memref<i64>, %arg96: memref<256xf32>, %arg97: memref<256xf32>, %arg98: memref<i64>, %arg99: memref<256xf32>, %arg100: memref<256xf32>, %arg101: memref<i64>, %arg102: memref<256xf32>, %arg103: memref<256xf32>, %arg104: memref<i64>, %arg105: memref<256xf32>, %arg106: memref<256xf32>, %arg107: memref<i64>, %arg108: memref<512xf32>, %arg109: memref<512xf32>, %arg110: memref<i64>, %arg111: memref<512xf32>, %arg112: memref<512xf32>, %arg113: memref<i64>, %arg114: memref<512xf32>, %arg115: memref<512xf32>, %arg116: memref<i64>, %arg117: memref<512xf32>, %arg118: memref<512xf32>, %arg119: memref<i64>, %arg120: memref<512xf32>, %arg121: memref<512xf32>, %arg122: memref<1x3x224x224xf32>) -> (memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>) {
    %0 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    %1 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%1) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %2 = call @Unknown0(%arg122) : (memref<1x3x224x224xf32>) -> memref<1x3x224x224xf16>
    %3 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %4 = memref.alloc() : memref<1x64x112x112xf16>
    lmhlo.convolution(%2, %3, %4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x3x224x224xf16>, memref<64x3x7x7xf16>, memref<1x64x112x112xf16>) -> ()
    %5:3 = call @BatchNormTrainingOp2(%4, %arg1, %arg0) : (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %6 = call @Unknown3(%5#0) : (memref<1x64x112x112xf16>) -> memref<1x64x112x112xf16>
    %7 = memref.alloc() : memref<1x64x56x56xf16>
    "lmhlo.reduce_window"(%6, %1, %7) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):
      %127 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg123, %arg124, %127) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%127, %arg125) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<1x64x112x112xf16>, memref<f16>, memref<1x64x56x56xf16>) -> ()
    %8 = call @Unknown4(%arg9) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %9 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%7, %8, %9) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %10:3 = call @BatchNormTrainingOp5(%9, %arg6, %arg5) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %11 = call @Unknown6(%10#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %12 = call @Unknown7(%arg10) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %13 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%11, %12, %13) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %14:3 = call @BatchNormTrainingOp8(%13, %arg8, %arg7) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %15 = call @Unknown9(%14#0, %7) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %16 = call @Unknown10(%arg15) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %17 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%15, %16, %17) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %18:3 = call @BatchNormTrainingOp11(%17, %arg12, %arg11) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %19 = call @Unknown12(%18#0) : (memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %20 = call @Unknown13(%arg16) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %21 = memref.alloc() : memref<1x64x56x56xf16>
    lmhlo.convolution(%19, %20, %21) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>) -> ()
    %22:3 = call @BatchNormTrainingOp14(%21, %arg14, %arg13) : (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> (memref<1x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %23 = call @Unknown15(%22#0, %15) : (memref<1x64x56x56xf16>, memref<1x64x56x56xf16>) -> memref<1x64x56x56xf16>
    %24 = call @Unknown16(%arg23) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %25 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%23, %24, %25) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>) -> ()
    %26:3 = call @BatchNormTrainingOp17(%25, %arg25, %arg24) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %27 = call @Unknown18(%arg21) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %28 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%23, %27, %28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %29:3 = call @BatchNormTrainingOp19(%28, %arg18, %arg17) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %30 = call @Unknown20(%29#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %31 = call @Unknown21(%arg22) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %32 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%30, %31, %32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %33:3 = call @BatchNormTrainingOp22(%32, %arg20, %arg19) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %34 = call @Unknown23(%33#0, %26#0) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %35 = call @Unknown24(%arg30) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %36 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%34, %35, %36) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %37:3 = call @BatchNormTrainingOp25(%36, %arg27, %arg26) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %38 = call @Unknown26(%37#0) : (memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %39 = call @Unknown27(%arg31) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %40 = memref.alloc() : memref<1x128x28x28xf16>
    lmhlo.convolution(%38, %39, %40) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>) -> ()
    %41:3 = call @BatchNormTrainingOp28(%40, %arg29, %arg28) : (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> (memref<1x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %42 = call @Unknown29(%41#0, %34) : (memref<1x128x28x28xf16>, memref<1x128x28x28xf16>) -> memref<1x128x28x28xf16>
    %43 = call @Unknown30(%arg38) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %44 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%42, %43, %44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>) -> ()
    %45:3 = call @BatchNormTrainingOp31(%44, %arg40, %arg39) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %46 = call @Unknown32(%arg36) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %47 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%42, %46, %47) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %48:3 = call @BatchNormTrainingOp33(%47, %arg33, %arg32) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %49 = call @Unknown34(%48#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %50 = call @Unknown35(%arg37) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %51 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%49, %50, %51) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %52:3 = call @BatchNormTrainingOp36(%51, %arg35, %arg34) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %53 = call @Unknown37(%52#0, %45#0) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %54 = call @Unknown38(%arg45) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %55 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%53, %54, %55) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %56:3 = call @BatchNormTrainingOp39(%55, %arg42, %arg41) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %57 = call @Unknown40(%56#0) : (memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %58 = call @Unknown41(%arg46) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %59 = memref.alloc() : memref<1x256x14x14xf16>
    lmhlo.convolution(%57, %58, %59) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>) -> ()
    %60:3 = call @BatchNormTrainingOp42(%59, %arg44, %arg43) : (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> (memref<1x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %61 = call @Unknown43(%60#0, %53) : (memref<1x256x14x14xf16>, memref<1x256x14x14xf16>) -> memref<1x256x14x14xf16>
    %62 = call @Unknown44(%arg53) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %63 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%61, %62, %63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>) -> ()
    %64:3 = call @BatchNormTrainingOp45(%63, %arg55, %arg54) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %65 = call @Unknown46(%arg51) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %66 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%61, %65, %66) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %67:3 = call @BatchNormTrainingOp47(%66, %arg48, %arg47) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %68 = call @Unknown48(%67#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %69 = call @Unknown49(%arg52) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %70 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%68, %69, %70) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %71:3 = call @BatchNormTrainingOp50(%70, %arg50, %arg49) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %72 = call @Unknown51(%71#0, %64#0) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %73 = call @Unknown52(%arg60) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %74 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%72, %73, %74) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %75:3 = call @BatchNormTrainingOp53(%74, %arg57, %arg56) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %76 = call @Unknown54(%75#0) : (memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %77 = call @Unknown55(%arg61) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %78 = memref.alloc() : memref<1x512x7x7xf16>
    lmhlo.convolution(%76, %77, %78) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>) -> ()
    %79:3 = call @BatchNormTrainingOp56(%78, %arg59, %arg58) : (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> (memref<1x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %80 = call @Unknown57(%79#0, %72) : (memref<1x512x7x7xf16>, memref<1x512x7x7xf16>) -> memref<1x512x7x7xf16>
    %81 = memref.alloc() : memref<1x512xf16>
    "lmhlo.reduce"(%80, %0, %81) ({
    ^bb0(%arg123: memref<f16>, %arg124: memref<f16>, %arg125: memref<f16>):
      "lmhlo.add"(%arg123, %arg124, %arg125) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<1x512x7x7xf16>, memref<f16>, memref<1x512xf16>) -> ()
    %82 = call @Unknown58(%81) : (memref<1x512xf16>) -> memref<1x512xf16>
    %83 = call @Unknown59(%arg4) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %84 = memref.alloc() : memref<512x1000xf16>
    "lmhlo.transpose"(%83, %84) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<1000x512xf16>, memref<512x1000xf16>) -> ()
    %85 = memref.alloc() : memref<1x1000xf16>
    "lmhlo.dot"(%82, %83, %85) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<1x512xf16>, memref<1000x512xf16>, memref<1x1000xf16>) -> ()
    %86 = call @Unknown60(%arg3, %85) : (memref<1000xf32>, memref<1x1000xf16>) -> memref<1x1000xf16>
    %87 = call @Unknown61(%5#1, %arg63) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %88 = call @Unknown62(%5#2, %arg64) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %89 = call @Unknown63(%10#1, %arg66) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %90 = call @Unknown64(%10#2, %arg67) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %91 = call @Unknown65(%14#1, %arg69) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %92 = call @Unknown66(%14#2, %arg70) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %93 = call @Unknown67(%18#1, %arg72) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %94 = call @Unknown68(%18#2, %arg73) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %95 = call @Unknown69(%22#1, %arg75) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %96 = call @Unknown70(%22#2, %arg76) : (memref<64xf32>, memref<64xf32>) -> memref<64xf32>
    %97 = call @Unknown71(%29#1, %arg78) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %98 = call @Unknown72(%29#2, %arg79) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %99 = call @Unknown73(%33#1, %arg81) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %100 = call @Unknown74(%33#2, %arg82) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %101 = call @Unknown75(%26#1, %arg84) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %102 = call @Unknown76(%26#2, %arg85) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %103 = call @Unknown77(%37#1, %arg87) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %104 = call @Unknown78(%37#2, %arg88) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %105 = call @Unknown79(%41#1, %arg90) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %106 = call @Unknown80(%41#2, %arg91) : (memref<128xf32>, memref<128xf32>) -> memref<128xf32>
    %107 = call @Unknown81(%48#1, %arg93) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %108 = call @Unknown82(%48#2, %arg94) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %109 = call @Unknown83(%52#1, %arg96) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %110 = call @Unknown84(%52#2, %arg97) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %111 = call @Unknown85(%45#1, %arg99) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %112 = call @Unknown86(%45#2, %arg100) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %113 = call @Unknown87(%56#1, %arg102) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %114 = call @Unknown88(%56#2, %arg103) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %115 = call @Unknown89(%60#1, %arg105) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %116 = call @Unknown90(%60#2, %arg106) : (memref<256xf32>, memref<256xf32>) -> memref<256xf32>
    %117 = call @Unknown91(%67#1, %arg108) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %118 = call @Unknown92(%67#2, %arg109) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %119 = call @Unknown93(%71#1, %arg111) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %120 = call @Unknown94(%71#2, %arg112) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %121 = call @Unknown95(%64#1, %arg114) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %122 = call @Unknown96(%64#2, %arg115) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %123 = call @Unknown97(%75#1, %arg117) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %124 = call @Unknown98(%75#2, %arg118) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %125 = call @Unknown99(%79#1, %arg120) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    %126 = call @Unknown100(%79#2, %arg121) : (memref<512xf32>, memref<512xf32>) -> memref<512xf32>
    return %86, %arg0, %arg1, %arg5, %arg6, %arg7, %arg8, %arg11, %arg12, %arg13, %arg14, %arg17, %arg18, %arg19, %arg20, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg32, %arg33, %arg34, %arg35, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg47, %arg48, %arg49, %arg50, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %3, %2, %4, %6, %7, %8, %9, %11, %12, %13, %15, %16, %17, %19, %20, %21, %23, %27, %28, %30, %31, %32, %24, %25, %34, %35, %36, %38, %39, %40, %42, %46, %47, %49, %50, %51, %43, %44, %53, %54, %55, %57, %58, %59, %61, %65, %66, %68, %69, %70, %62, %63, %72, %73, %74, %76, %77, %78, %80, %82, %84 : memref<1x1000xf16>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<64x3x7x7xf16>, memref<1x3x224x224xf16>, memref<1x64x112x112xf16>, memref<1x64x112x112xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<64x64x3x3xf16>, memref<1x64x56x56xf16>, memref<1x64x56x56xf16>, memref<128x64x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<128x64x1x1xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<128x128x3x3xf16>, memref<1x128x28x28xf16>, memref<1x128x28x28xf16>, memref<256x128x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<256x128x1x1xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<256x256x3x3xf16>, memref<1x256x14x14xf16>, memref<1x256x14x14xf16>, memref<512x256x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<512x256x1x1xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<512x512x3x3xf16>, memref<1x512x7x7xf16>, memref<1x512x7x7xf16>, memref<1x512xf16>, memref<512x1000xf16>
  }
}

