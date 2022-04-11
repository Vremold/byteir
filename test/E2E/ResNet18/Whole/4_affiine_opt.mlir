// RUN: byteir-opt %s -affine-opt | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map6 = affine_map<() -> ()>
module @IrToMhlo.2452 {
  func private @Unknown0(%arg0: memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4x3x224x224xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x3x224x224xf32>) outs(%0 : memref<4x3x224x224xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x3x224x224xf16>
  }
  func private @Unknown1(%arg0: memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x3x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf32>) outs(%0 : memref<64x3x7x7xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x3x7x7xf16>
  }
  func private @BatchNormTrainingOp2(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x112x112xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x112x112xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %4 : memref<4x64x112x112xf16>
  }
  func private @Unknown3(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown4(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown5(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown6(%arg0: memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf32>) outs(%0 : memref<64x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<64x64x3x3xf16>
  }
  func private @Unknown7(%arg0: memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf32>) outs(%0 : memref<128x64x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x64x1x1xf16>
  }
  func private @Unknown8(%arg0: memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf32>) outs(%0 : memref<128x64x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x64x3x3xf16>
  }
  func private @Unknown9(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown10(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown11(%arg0: memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf32>) outs(%0 : memref<128x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<128x128x3x3xf16>
  }
  func private @Unknown12(%arg0: memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf32>) outs(%0 : memref<256x128x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x128x1x1xf16>
  }
  func private @Unknown13(%arg0: memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf32>) outs(%0 : memref<256x128x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x128x3x3xf16>
  }
  func private @Unknown14(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown15(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown16(%arg0: memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf32>) outs(%0 : memref<256x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<256x256x3x3xf16>
  }
  func private @Unknown17(%arg0: memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x1x1xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf32>) outs(%0 : memref<512x256x1x1xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x256x1x1xf16>
  }
  func private @Unknown18(%arg0: memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf32>) outs(%0 : memref<512x256x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x256x3x3xf16>
  }
  func private @Unknown19(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown20(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown21(%arg0: memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf32>) outs(%0 : memref<512x512x3x3xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<512x512x3x3xf16>
  }
  func private @Unknown22(%arg0: memref<4x1000xf32>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -2.500000e-01 : f32
    %0 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x1000xf32>) outs(%0 : memref<4x1000xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.mulf %arg1, %cst : f32
      %2 = arith.truncf %1 : f32 to f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x1000xf16>
  }
  func private @Unknown23(%arg0: memref<1000x512xf32>) -> memref<1000x512xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf32>) outs(%0 : memref<1000x512xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1000x512xf16>
  }
  func private @Unknown24(%arg0: memref<1000xf32>) -> memref<1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf16>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg0 : memref<1000xf32>) outs(%0 : memref<1000xf16>) {
    ^bb0(%arg1: f32, %arg2: f16):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      linalg.yield %1 : f16
    }
    return %0 : memref<1000xf16>
  }
  func private @Unknown25(%arg0: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x112x112xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x112x112xf16>) outs(%0 : memref<4x64x112x112xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x64x112x112xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x112x112xf16>) outs(%1 : memref<4x64x112x112xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
  }
  func private @BatchNormTrainingOp26(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %4 : memref<4x64x56x56xf16>
  }
  func private @Unknown27(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%1 : memref<4x64x56x56xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp28(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %4 : memref<4x64x56x56xf16>
  }
  func private @Unknown29(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%1 : memref<4x64x56x56xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp30(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %4 : memref<4x64x56x56xf16>
  }
  func private @Unknown31(%arg0: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x64x56x56xf16>) outs(%1 : memref<4x64x56x56xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp32(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) -> memref<4x64x56x56xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    %2 = memref.alloc() : memref<64xf32>
    %3 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %4 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %4 : memref<4x64x56x56xf16>
  }
  func private @Unknown33(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x64x56x56xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%1 : memref<4x64x56x56xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
  }
  func private @BatchNormTrainingOp34(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %4 : memref<4x128x28x28xf16>
  }
  func private @BatchNormTrainingOp35(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %4 : memref<4x128x28x28xf16>
  }
  func private @Unknown36(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%1 : memref<4x128x28x28xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp37(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %4 : memref<4x128x28x28xf16>
  }
  func private @Unknown38(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%1 : memref<4x128x28x28xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp39(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %4 : memref<4x128x28x28xf16>
  }
  func private @Unknown40(%arg0: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x128x28x28xf16>) outs(%1 : memref<4x128x28x28xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp41(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<4x128x28x28xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    %2 = memref.alloc() : memref<128xf32>
    %3 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %4 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %4 : memref<4x128x28x28xf16>
  }
  func private @Unknown42(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x128x28x28xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%1 : memref<4x128x28x28xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
  }
  func private @BatchNormTrainingOp43(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %4 : memref<4x256x14x14xf16>
  }
  func private @BatchNormTrainingOp44(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %4 : memref<4x256x14x14xf16>
  }
  func private @Unknown45(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%1 : memref<4x256x14x14xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp46(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %4 : memref<4x256x14x14xf16>
  }
  func private @Unknown47(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%1 : memref<4x256x14x14xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp48(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %4 : memref<4x256x14x14xf16>
  }
  func private @Unknown49(%arg0: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x256x14x14xf16>) outs(%1 : memref<4x256x14x14xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp50(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<256xf32>) -> memref<4x256x14x14xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    %2 = memref.alloc() : memref<256xf32>
    %3 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %4 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %4 : memref<4x256x14x14xf16>
  }
  func private @Unknown51(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x256x14x14xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%1 : memref<4x256x14x14xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
  }
  func private @BatchNormTrainingOp52(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %4 : memref<4x512x7x7xf16>
  }
  func private @BatchNormTrainingOp53(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %4 : memref<4x512x7x7xf16>
  }
  func private @Unknown54(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%1 : memref<4x512x7x7xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp55(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %4 : memref<4x512x7x7xf16>
  }
  func private @Unknown56(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%1 : memref<4x512x7x7xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp57(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %4 : memref<4x512x7x7xf16>
  }
  func private @Unknown58(%arg0: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<4x512x7x7xf16>) outs(%1 : memref<4x512x7x7xi1>) {
    ^bb0(%arg1: f16, %arg2: i1):  // no predecessors
      %2 = arith.maxf %arg1, %cst : f16
      %3 = arith.cmpf ogt, %2, %cst : f16
      linalg.yield %3 : i1
    }
    return %0, %1 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @BatchNormTrainingOp59(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<512xf32>) -> memref<4x512x7x7xf16> attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"} {
    %0 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    %2 = memref.alloc() : memref<512xf32>
    %3 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_training"(%0, %arg1, %arg2, %1, %2, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %4 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%1, %4) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %4 : memref<4x512x7x7xf16>
  }
  func private @Unknown60(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      linalg.yield %3 : f16
    }
    %1 = memref.alloc() : memref<4x512x7x7xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%1 : memref<4x512x7x7xi1>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: i1):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f16
      %3 = arith.maxf %2, %cst : f16
      %4 = arith.cmpf ogt, %3, %cst : f16
      linalg.yield %4 : i1
    }
    return %0, %1 : memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
  }
  func private @Unknown61(%arg0: memref<4x512xf16>) -> memref<4x512xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 2.040100e-02 : f16
    %0 = memref.alloc() : memref<4x512xf16>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x512xf16>) outs(%0 : memref<4x512xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %1 = arith.mulf %arg1, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x512xf16>
  }
  func private @Unknown62(%arg0: memref<1000xf16>, %arg1: memref<4x1000xf16>) -> memref<4x1000xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x1000xf16>, memref<1000xf16>) outs(%0 : memref<4x1000xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = arith.addf %arg2, %arg3 : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x1000xf16>
  }
  func private @Unknown63(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x1000xf16>, memref<4xf16>) outs(%0 : memref<4x1000xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.subf %arg2, %arg3 : f16
      linalg.yield %2 : f16
    }
    %1 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x1000xf16>, memref<4xf16>) outs(%1 : memref<4x1000xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %2 = arith.subf %arg2, %arg3 : f16
      %3 = math.exp %2 : f16
      linalg.yield %3 : f16
    }
    return %0, %1 : memref<4x1000xf16>, memref<4x1000xf16>
  }
  func private @Unknown64(%arg0: memref<4xf16>) -> memref<4xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4xf16>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg0 : memref<4xf16>) outs(%0 : memref<4xf16>) {
    ^bb0(%arg1: f16, %arg2: f16):  // no predecessors
      %1 = math.log %arg1 : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4xf16>
  }
  func private @Unknown65(%arg0: memref<4xf16>, %arg1: memref<4x1000xf16>, %arg2: memref<4xf16>, %arg3: memref<4x1000xf16>, %arg4: memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4x1000xf16>
    linalg.generic {indexing_maps = [#map1, #map1, #map4, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg1, %arg0, %arg2 : memref<4x1000xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4xf16>) outs(%0 : memref<4x1000xf16>) {
    ^bb0(%arg5: f16, %arg6: f16, %arg7: f16, %arg8: f16, %arg9: f16):  // no predecessors
      %3 = arith.subf %arg6, %arg7 : f16
      %4 = math.exp %3 : f16
      %5 = arith.mulf %4, %arg8 : f16
      %6 = arith.subf %arg5, %5 : f16
      linalg.yield %6 : f16
    }
    %1 = memref.alloc() : memref<4x1000xf32>
    linalg.generic {indexing_maps = [#map1, #map4, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0, %arg4 : memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf32>) outs(%1 : memref<4x1000xf32>) {
    ^bb0(%arg5: f16, %arg6: f16, %arg7: f32, %arg8: f32):  // no predecessors
      %3 = arith.subf %arg5, %arg6 : f16
      %4 = arith.extf %3 : f16 to f32
      %5 = arith.mulf %4, %arg7 : f32
      linalg.yield %5 : f32
    }
    %2 = memref.alloc() : memref<4x1000xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map4, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg1, %arg0, %arg2 : memref<4x1000xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4xf16>) outs(%2 : memref<4x1000xf32>) {
    ^bb0(%arg5: f16, %arg6: f16, %arg7: f16, %arg8: f16, %arg9: f32):  // no predecessors
      %3 = arith.subf %arg6, %arg7 : f16
      %4 = math.exp %3 : f16
      %5 = arith.mulf %4, %arg8 : f16
      %6 = arith.subf %arg5, %5 : f16
      %7 = arith.extf %6 : f16 to f32
      linalg.yield %7 : f32
    }
    return %0, %1, %2 : memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>
  }
  func private @Unknown66(%arg0: memref<4x512xf16>, %arg1: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 4.900000e+01 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map5, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %arg0 : memref<4x512x7x7xi1>, memref<4x512xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = arith.divf %arg3, %cst_0 : f16
      %2 = select %arg2, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp67(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<4x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp68(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %2 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp69(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown70(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp71(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<4x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp72(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %2 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp73(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown74(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>, %arg2: memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp75(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<4x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp76(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<512x512x3x3xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x512x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,512,512]{1,0,2,3}"} : (memref<3x3x512x512xf16>, memref<3x3x512x512xf16>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x512x512xf16>, memref<4x512x7x7xf16>) -> ()
    return %2 : memref<4x512x7x7xf16>
  }
  func private @ConvBackwardFilterOp77(%arg0: memref<4x512x7x7xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x512x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<3x3x512x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x512x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,512,3,3]{0,1,3,2}"} : (memref<3x3x512x512xf16>, memref<512x512x3x3xf16>) -> ()
    return %1 : memref<512x512x3x3xf16>
  }
  func private @Unknown78(%arg0: memref<4x512x7x7xi1>, %arg1: memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x512x7x7xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) outs(%0 : memref<4x512x7x7xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x512x7x7xf16>
  }
  func private @BatchNormGradOp79(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<4x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp80(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<512x256x3x3xf16>, memref<3x3x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x512xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,512]{1,0,2,3}"} : (memref<3x3x256x512xf16>, memref<3x3x256x512xf16>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<3x3x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %2 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp81(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<3x3x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,3,3]{0,1,3,2}"} : (memref<3x3x256x512xf16>, memref<512x256x3x3xf16>) -> ()
    return %1 : memref<512x256x3x3xf16>
  }
  func private @BatchNormGradOp82(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512xf32>, %arg2: memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<512xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512xf32>} : (memref<512xf32>) -> ()
    %1 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %2 = memref.alloc() : memref<4x512x7x7xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf32>) -> ()
    %3 = memref.alloc() : memref<4x512x7x7xf32>
    %4 = memref.alloc() : memref<512xf32>
    %5 = memref.alloc() : memref<512xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf32>, memref<4x512x7x7xf32>, memref<512xf32>, memref<512xf32>) -> ()
    %6 = memref.alloc() : memref<4x512x7x7xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x512x7x7xf32>, memref<4x512x7x7xf16>) -> ()
    return %6, %4, %5 : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
  }
  func private @ConvBackwardDataOp83(%arg0: memref<4x512x7x7xf16>, %arg1: memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,256,512]{1,0,2,3}"} : (memref<512x256x1x1xf16>, memref<1x1x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<1x1x256x512xf16>, memref<4x256x14x14xf16>) -> ()
    return %1 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp84(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x256x512xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<1x1x256x512xf16>) -> ()
    %1 = memref.alloc() : memref<512x256x1x1xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[512,256,1,1]{0,1,3,2}"} : (memref<1x1x256x512xf16>, memref<512x256x1x1xf16>) -> ()
    return %1 : memref<512x256x1x1xf16>
  }
  func private @Unknown85(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp86(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<4x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp87(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %2 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp88(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown89(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp90(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<4x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp91(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %2 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp92(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown93(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>, %arg2: memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp94(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<4x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp95(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<256x256x3x3xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x256x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,256,256]{1,0,2,3}"} : (memref<3x3x256x256xf16>, memref<3x3x256x256xf16>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x256x256xf16>, memref<4x256x14x14xf16>) -> ()
    return %2 : memref<4x256x14x14xf16>
  }
  func private @ConvBackwardFilterOp96(%arg0: memref<4x256x14x14xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x256x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<3x3x256x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x256x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,256,3,3]{0,1,3,2}"} : (memref<3x3x256x256xf16>, memref<256x256x3x3xf16>) -> ()
    return %1 : memref<256x256x3x3xf16>
  }
  func private @Unknown97(%arg0: memref<4x256x14x14xi1>, %arg1: memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x256x14x14xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) outs(%0 : memref<4x256x14x14xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x256x14x14xf16>
  }
  func private @BatchNormGradOp98(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<4x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp99(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<256x128x3x3xf16>, memref<3x3x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x256xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,256]{1,0,2,3}"} : (memref<3x3x128x256xf16>, memref<3x3x128x256xf16>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<3x3x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %2 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp100(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<3x3x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,3,3]{0,1,3,2}"} : (memref<3x3x128x256xf16>, memref<256x128x3x3xf16>) -> ()
    return %1 : memref<256x128x3x3xf16>
  }
  func private @BatchNormGradOp101(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256xf32>, %arg2: memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<256xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<256xf32>} : (memref<256xf32>) -> ()
    %1 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %2 = memref.alloc() : memref<4x256x14x14xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf32>) -> ()
    %3 = memref.alloc() : memref<4x256x14x14xf32>
    %4 = memref.alloc() : memref<256xf32>
    %5 = memref.alloc() : memref<256xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf32>, memref<4x256x14x14xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %6 = memref.alloc() : memref<4x256x14x14xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x256x14x14xf32>, memref<4x256x14x14xf16>) -> ()
    return %6, %4, %5 : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
  }
  func private @ConvBackwardDataOp102(%arg0: memref<4x256x14x14xf16>, %arg1: memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,128,256]{1,0,2,3}"} : (memref<256x128x1x1xf16>, memref<1x1x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<1x1x128x256xf16>, memref<4x128x28x28xf16>) -> ()
    return %1 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp103(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x128x256xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<1x1x128x256xf16>) -> ()
    %1 = memref.alloc() : memref<256x128x1x1xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[256,128,1,1]{0,1,3,2}"} : (memref<1x1x128x256xf16>, memref<256x128x1x1xf16>) -> ()
    return %1 : memref<256x128x1x1xf16>
  }
  func private @Unknown104(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp105(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<4x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp106(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %2 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp107(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown108(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp109(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<4x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp110(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %2 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp111(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown112(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>, %arg2: memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp113(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<4x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp114(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<128x128x3x3xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x128x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,128,128]{1,0,2,3}"} : (memref<3x3x128x128xf16>, memref<3x3x128x128xf16>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x128x128xf16>, memref<4x128x28x28xf16>) -> ()
    return %2 : memref<4x128x28x28xf16>
  }
  func private @ConvBackwardFilterOp115(%arg0: memref<4x128x28x28xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x128x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<3x3x128x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x128x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,128,3,3]{0,1,3,2}"} : (memref<3x3x128x128xf16>, memref<128x128x3x3xf16>) -> ()
    return %1 : memref<128x128x3x3xf16>
  }
  func private @Unknown116(%arg0: memref<4x128x28x28xi1>, %arg1: memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x128x28x28xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) outs(%0 : memref<4x128x28x28xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x128x28x28xf16>
  }
  func private @BatchNormGradOp117(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<4x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp118(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<128x64x3x3xf16>, memref<3x3x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x128xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,128]{1,0,2,3}"} : (memref<3x3x64x128xf16>, memref<3x3x64x128xf16>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<3x3x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %2 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp119(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<3x3x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,3,3]{0,1,3,2}"} : (memref<3x3x64x128xf16>, memref<128x64x3x3xf16>) -> ()
    return %1 : memref<128x64x3x3xf16>
  }
  func private @BatchNormGradOp120(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128xf32>, %arg2: memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128xf32>} : (memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %2 = memref.alloc() : memref<4x128x28x28xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf32>) -> ()
    %3 = memref.alloc() : memref<4x128x28x28xf32>
    %4 = memref.alloc() : memref<128xf32>
    %5 = memref.alloc() : memref<128xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf32>, memref<4x128x28x28xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %6 = memref.alloc() : memref<4x128x28x28xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x128x28x28xf32>, memref<4x128x28x28xf16>) -> ()
    return %6, %4, %5 : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
  }
  func private @ConvBackwardDataOp121(%arg0: memref<4x128x28x28xf16>, %arg1: memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[1,1,64,128]{1,0,2,3}"} : (memref<128x64x1x1xf16>, memref<1x1x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<1x1x64x128xf16>, memref<4x64x56x56xf16>) -> ()
    return %1 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp122(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<1x1x64x128xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<1x1x64x128xf16>) -> ()
    %1 = memref.alloc() : memref<128x64x1x1xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[128,64,1,1]{0,1,3,2}"} : (memref<1x1x64x128xf16>, memref<128x64x1x1xf16>) -> ()
    return %1 : memref<128x64x1x1xf16>
  }
  func private @Unknown123(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp124(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<4x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp125(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %2 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp126(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown127(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp128(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<4x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp129(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %2 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp130(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown131(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>, %arg2: memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg3: i1, %arg4: f16, %arg5: f16, %arg6: f16):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f16
      %2 = select %arg3, %1, %cst : f16
      linalg.yield %2 : f16
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp132(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<4x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp133(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %2 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp134(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown135(%arg0: memref<4x64x56x56xi1>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @BatchNormGradOp136(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf32>) -> ()
    %3 = memref.alloc() : memref<4x64x56x56xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf32>, memref<4x64x56x56xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x64x56x56xf32>, memref<4x64x56x56xf16>) -> ()
    return %6, %4, %5 : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardDataOp137(%arg0: memref<4x64x56x56xf16>, %arg1: memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.transpose"(%arg1, %0) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<64x64x3x3xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<3x3x64x64xf16>
    "lmhlo.reverse"(%0, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>, xla_shape = "f16[3,3,64,64]{1,0,2,3}"} : (memref<3x3x64x64xf16>, memref<3x3x64x64xf16>) -> ()
    %2 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%arg0, %1, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<3x3x64x64xf16>, memref<4x64x56x56xf16>) -> ()
    return %2 : memref<4x64x56x56xf16>
  }
  func private @ConvBackwardFilterOp138(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<3x3x64x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<3x3x64x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x64x3x3xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,64,3,3]{0,1,3,2}"} : (memref<3x3x64x64xf16>, memref<64x64x3x3xf16>) -> ()
    return %1 : memref<64x64x3x3xf16>
  }
  func private @Unknown139(%arg0: memref<4x64x56x56xf16>, %arg1: memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<4x64x56x56xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) outs(%0 : memref<4x64x56x56xf16>) {
    ^bb0(%arg2: f16, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = arith.addf %arg2, %arg3 : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x64x56x56xf16>
  }
  func private @Unknown140(%arg0: memref<4x64x112x112xi1>, %arg1: memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = memref.alloc() : memref<4x64x112x112xf16>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) outs(%0 : memref<4x64x112x112xf16>) {
    ^bb0(%arg2: i1, %arg3: f16, %arg4: f16):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f16
      linalg.yield %1 : f16
    }
    return %0 : memref<4x64x112x112xf16>
  }
  func private @BatchNormGradOp141(%arg0: memref<4x64x112x112xf16>, %arg1: memref<64xf32>, %arg2: memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) attributes {__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"} {
    %0 = memref.alloc() : memref<64xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<64xf32>} : (memref<64xf32>) -> ()
    %1 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg0, %1) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %2 = memref.alloc() : memref<4x64x112x112xf32>
    "lmhlo.convert"(%arg2, %2) : (memref<4x64x112x112xf16>, memref<4x64x112x112xf32>) -> ()
    %3 = memref.alloc() : memref<4x64x112x112xf32>
    %4 = memref.alloc() : memref<64xf32>
    %5 = memref.alloc() : memref<64xf32>
    "lmhlo.batch_norm_grad"(%1, %arg1, %0, %0, %2, %3, %4, %5) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf32>, memref<4x64x112x112xf32>, memref<64xf32>, memref<64xf32>) -> ()
    %6 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.convert"(%3, %6) : (memref<4x64x112x112xf32>, memref<4x64x112x112xf16>) -> ()
    return %6, %4, %5 : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
  }
  func private @ConvBackwardFilterOp142(%arg0: memref<4x3x224x224xf16>, %arg1: memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16> attributes {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<3> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"} {
    %0 = memref.alloc() : memref<7x7x3x64xf16>
    lmhlo.convolution(%arg0, %arg1, %0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<7x7x3x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x3x7x7xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>, xla_shape = "f16[64,3,7,7]{0,1,3,2}"} : (memref<7x7x3x64xf16>, memref<64x3x7x7xf16>) -> ()
    return %1 : memref<64x3x7x7xf16>
  }
  func private @Unknown143(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 4.000000e+00 : f32
    %0 = memref.alloc() : memref<f32>
    linalg.generic {indexing_maps = [#map6, #map6], iterator_types = []} ins(%arg0 : memref<f32>) outs(%0 : memref<f32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %1 = arith.negf %arg1 : f32
      %2 = arith.divf %1, %cst : f32
      linalg.yield %2 : f32
    }
    return %0 : memref<f32>
  }
  func private @Unknown144(%arg0: memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x3x7x7xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x3x7x7xf16>) outs(%0 : memref<64x3x7x7xf32>) attrs =  {xla_shape = "f32[64,3,7,7]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x3x7x7xf32>
  }
  func private @Unknown145(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown146(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown147(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown148(%arg0: memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<64x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<64x64x3x3xf16>) outs(%0 : memref<64x64x3x3xf32>) attrs =  {xla_shape = "f32[64,64,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<64x64x3x3xf32>
  }
  func private @Unknown149(%arg0: memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x3x3xf16>) outs(%0 : memref<128x64x3x3xf32>) attrs =  {xla_shape = "f32[128,64,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x64x3x3xf32>
  }
  func private @Unknown150(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown151(%arg0: memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x64x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x64x1x1xf16>) outs(%0 : memref<128x64x1x1xf32>) attrs =  {xla_shape = "f32[128,64,1,1]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x64x1x1xf32>
  }
  func private @Unknown152(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown153(%arg0: memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<128x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<128x128x3x3xf16>) outs(%0 : memref<128x128x3x3xf32>) attrs =  {xla_shape = "f32[128,128,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128x3x3xf32>
  }
  func private @Unknown154(%arg0: memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x3x3xf16>) outs(%0 : memref<256x128x3x3xf32>) attrs =  {xla_shape = "f32[256,128,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x128x3x3xf32>
  }
  func private @Unknown155(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown156(%arg0: memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x128x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x128x1x1xf16>) outs(%0 : memref<256x128x1x1xf32>) attrs =  {xla_shape = "f32[256,128,1,1]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x128x1x1xf32>
  }
  func private @Unknown157(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown158(%arg0: memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<256x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<256x256x3x3xf16>) outs(%0 : memref<256x256x3x3xf32>) attrs =  {xla_shape = "f32[256,256,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x256x3x3xf32>
  }
  func private @Unknown159(%arg0: memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x3x3xf16>) outs(%0 : memref<512x256x3x3xf32>) attrs =  {xla_shape = "f32[512,256,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x256x3x3xf32>
  }
  func private @Unknown160(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown161(%arg0: memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x256x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x256x1x1xf16>) outs(%0 : memref<512x256x1x1xf32>) attrs =  {xla_shape = "f32[512,256,1,1]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x256x1x1xf32>
  }
  func private @Unknown162(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @Unknown163(%arg0: memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<512x512x3x3xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<512x512x3x3xf16>) outs(%0 : memref<512x512x3x3xf32>) attrs =  {xla_shape = "f32[512,512,3,3]{0,1,3,2}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<512x512x3x3xf32>
  }
  func private @MatmulOp164(%arg0: memref<4x512xf16>, %arg1: memref<4x1000xf16>) -> memref<1000x512xf16> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<512x1000xf16>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512xf16>, memref<4x1000xf16>, memref<512x1000xf16>) -> ()
    %1 = memref.alloc() : memref<1000x512xf16>
    "lmhlo.transpose"(%0, %1) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f16[1000,512]{0,1}"} : (memref<512x1000xf16>, memref<1000x512xf16>) -> ()
    return %1 : memref<1000x512xf16>
  }
  func private @Unknown165(%arg0: memref<1000x512xf16>) -> memref<1000x512xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000x512xf32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1000x512xf16>) outs(%0 : memref<1000x512xf32>) attrs =  {xla_shape = "f32[1000,512]{0,1}"} {
    ^bb0(%arg1: f16, %arg2: f32):  // no predecessors
      %1 = arith.extf %arg1 : f16 to f32
      linalg.yield %1 : f32
    }
    return %0 : memref<1000x512xf32>
  }
  func private @Unknown166(%arg0: memref<1000xf32>) -> memref<1000xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() : memref<1000xf32>
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg0 : memref<1000xf32>) outs(%0 : memref<1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %1 = arith.truncf %arg1 : f32 to f16
      %2 = arith.extf %1 : f16 to f32
      linalg.yield %2 : f32
    }
    return %0 : memref<1000xf32>
  }
  func @main(%arg0: memref<4x3x224x224xf32>, %arg1: memref<4x1000xf32>, %arg2: memref<64x3x7x7xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64x64x3x3xf32>, %arg8: memref<64xf32>, %arg9: memref<64xf32>, %arg10: memref<64xf32>, %arg11: memref<64xf32>, %arg12: memref<64x64x3x3xf32>, %arg13: memref<64xf32>, %arg14: memref<64xf32>, %arg15: memref<64xf32>, %arg16: memref<64xf32>, %arg17: memref<64x64x3x3xf32>, %arg18: memref<64xf32>, %arg19: memref<64xf32>, %arg20: memref<64xf32>, %arg21: memref<64xf32>, %arg22: memref<64x64x3x3xf32>, %arg23: memref<64xf32>, %arg24: memref<64xf32>, %arg25: memref<64xf32>, %arg26: memref<64xf32>, %arg27: memref<128x64x3x3xf32>, %arg28: memref<128xf32>, %arg29: memref<128xf32>, %arg30: memref<128xf32>, %arg31: memref<128xf32>, %arg32: memref<128x128x3x3xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<128xf32>, %arg37: memref<128x64x1x1xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128x3x3xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<128xf32>, %arg47: memref<128x128x3x3xf32>, %arg48: memref<128xf32>, %arg49: memref<128xf32>, %arg50: memref<128xf32>, %arg51: memref<128xf32>, %arg52: memref<256x128x3x3xf32>, %arg53: memref<256xf32>, %arg54: memref<256xf32>, %arg55: memref<256xf32>, %arg56: memref<256xf32>, %arg57: memref<256x256x3x3xf32>, %arg58: memref<256xf32>, %arg59: memref<256xf32>, %arg60: memref<256xf32>, %arg61: memref<256xf32>, %arg62: memref<256x128x1x1xf32>, %arg63: memref<256xf32>, %arg64: memref<256xf32>, %arg65: memref<256xf32>, %arg66: memref<256xf32>, %arg67: memref<256x256x3x3xf32>, %arg68: memref<256xf32>, %arg69: memref<256xf32>, %arg70: memref<256xf32>, %arg71: memref<256xf32>, %arg72: memref<256x256x3x3xf32>, %arg73: memref<256xf32>, %arg74: memref<256xf32>, %arg75: memref<256xf32>, %arg76: memref<256xf32>, %arg77: memref<512x256x3x3xf32>, %arg78: memref<512xf32>, %arg79: memref<512xf32>, %arg80: memref<512xf32>, %arg81: memref<512xf32>, %arg82: memref<512x512x3x3xf32>, %arg83: memref<512xf32>, %arg84: memref<512xf32>, %arg85: memref<512xf32>, %arg86: memref<512xf32>, %arg87: memref<512x256x1x1xf32>, %arg88: memref<512xf32>, %arg89: memref<512xf32>, %arg90: memref<512xf32>, %arg91: memref<512xf32>, %arg92: memref<512x512x3x3xf32>, %arg93: memref<512xf32>, %arg94: memref<512xf32>, %arg95: memref<512xf32>, %arg96: memref<512xf32>, %arg97: memref<512x512x3x3xf32>, %arg98: memref<512xf32>, %arg99: memref<512xf32>, %arg100: memref<512xf32>, %arg101: memref<512xf32>, %arg102: memref<1000x512xf32>, %arg103: memref<1000xf32>) -> (memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>) {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<f16>} : (memref<f16>) -> ()
    %2 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%2) {value = dense<0xFC00> : tensor<f16>} : (memref<f16>) -> ()
    %3 = call @Unknown0(%arg0) : (memref<4x3x224x224xf32>) -> memref<4x3x224x224xf16>
    %4 = call @Unknown1(%arg2) : (memref<64x3x7x7xf32>) -> memref<64x3x7x7xf16>
    %5 = memref.alloc() : memref<4x64x112x112xf16>
    lmhlo.convolution(%3, %4, %5) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>) -> ()
    %6 = call @BatchNormTrainingOp2(%5, %arg3, %arg4) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x112x112xf16>
    %7 = call @Unknown3(%arg7) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %8 = call @Unknown4(%arg12) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %9 = call @Unknown5(%arg17) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %10 = call @Unknown6(%arg22) : (memref<64x64x3x3xf32>) -> memref<64x64x3x3xf16>
    %11 = call @Unknown7(%arg37) : (memref<128x64x1x1xf32>) -> memref<128x64x1x1xf16>
    %12 = call @Unknown8(%arg27) : (memref<128x64x3x3xf32>) -> memref<128x64x3x3xf16>
    %13 = call @Unknown9(%arg32) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %14 = call @Unknown10(%arg42) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %15 = call @Unknown11(%arg47) : (memref<128x128x3x3xf32>) -> memref<128x128x3x3xf16>
    %16 = call @Unknown12(%arg62) : (memref<256x128x1x1xf32>) -> memref<256x128x1x1xf16>
    %17 = call @Unknown13(%arg52) : (memref<256x128x3x3xf32>) -> memref<256x128x3x3xf16>
    %18 = call @Unknown14(%arg57) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %19 = call @Unknown15(%arg67) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %20 = call @Unknown16(%arg72) : (memref<256x256x3x3xf32>) -> memref<256x256x3x3xf16>
    %21 = call @Unknown17(%arg87) : (memref<512x256x1x1xf32>) -> memref<512x256x1x1xf16>
    %22 = call @Unknown18(%arg77) : (memref<512x256x3x3xf32>) -> memref<512x256x3x3xf16>
    %23 = call @Unknown19(%arg82) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %24 = call @Unknown20(%arg92) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %25 = call @Unknown21(%arg97) : (memref<512x512x3x3xf32>) -> memref<512x512x3x3xf16>
    %26 = call @Unknown22(%arg1) : (memref<4x1000xf32>) -> memref<4x1000xf16>
    %27 = call @Unknown23(%arg102) : (memref<1000x512xf32>) -> memref<1000x512xf16>
    %28 = call @Unknown24(%arg103) : (memref<1000xf32>) -> memref<1000xf16>
    %29 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%26, %1, %29) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %30:2 = call @Unknown25(%6) : (memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<4x64x112x112xi1>)
    %31 = memref.alloc() : memref<4x64x56x56xf16>
    "lmhlo.reduce_window"(%30#0, %2, %31) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      %200 = memref.alloc() : memref<f16>
      "lmhlo.maximum"(%arg104, %arg105, %200) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.copy"(%200, %arg106) : (memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<f16>, memref<4x64x56x56xf16>) -> ()
    %32 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%31, %7, %32) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %33 = call @BatchNormTrainingOp26(%32, %arg8, %arg9) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %34:2 = call @Unknown27(%33) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %35 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%34#0, %8, %35) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %36 = call @BatchNormTrainingOp28(%35, %arg13, %arg14) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %37:2 = call @Unknown29(%36, %31) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %38 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%37#0, %9, %38) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %39 = call @BatchNormTrainingOp30(%38, %arg18, %arg19) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %40:2 = call @Unknown31(%39) : (memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %41 = memref.alloc() : memref<4x64x56x56xf16>
    lmhlo.convolution(%40#0, %10, %41) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>) -> ()
    %42 = call @BatchNormTrainingOp32(%41, %arg23, %arg24) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>) -> memref<4x64x56x56xf16>
    %43:2 = call @Unknown33(%42, %37#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<4x64x56x56xi1>)
    %44 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%43#0, %11, %44) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>) -> ()
    %45 = call @BatchNormTrainingOp34(%44, %arg38, %arg39) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %46 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%43#0, %12, %46) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %47 = call @BatchNormTrainingOp35(%46, %arg28, %arg29) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %48:2 = call @Unknown36(%47) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %49 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%48#0, %13, %49) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %50 = call @BatchNormTrainingOp37(%49, %arg33, %arg34) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %51:2 = call @Unknown38(%50, %45) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %52 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%51#0, %14, %52) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %53 = call @BatchNormTrainingOp39(%52, %arg43, %arg44) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %54:2 = call @Unknown40(%53) : (memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %55 = memref.alloc() : memref<4x128x28x28xf16>
    lmhlo.convolution(%54#0, %15, %55) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>) -> ()
    %56 = call @BatchNormTrainingOp41(%55, %arg48, %arg49) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>) -> memref<4x128x28x28xf16>
    %57:2 = call @Unknown42(%56, %51#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<4x128x28x28xi1>)
    %58 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%57#0, %16, %58) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>) -> ()
    %59 = call @BatchNormTrainingOp43(%58, %arg63, %arg64) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %60 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%57#0, %17, %60) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %61 = call @BatchNormTrainingOp44(%60, %arg53, %arg54) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %62:2 = call @Unknown45(%61) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %63 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%62#0, %18, %63) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %64 = call @BatchNormTrainingOp46(%63, %arg58, %arg59) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %65:2 = call @Unknown47(%64, %59) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %66 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%65#0, %19, %66) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %67 = call @BatchNormTrainingOp48(%66, %arg68, %arg69) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %68:2 = call @Unknown49(%67) : (memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %69 = memref.alloc() : memref<4x256x14x14xf16>
    lmhlo.convolution(%68#0, %20, %69) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>) -> ()
    %70 = call @BatchNormTrainingOp50(%69, %arg73, %arg74) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>) -> memref<4x256x14x14xf16>
    %71:2 = call @Unknown51(%70, %65#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<4x256x14x14xi1>)
    %72 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%71#0, %21, %72) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>) -> ()
    %73 = call @BatchNormTrainingOp52(%72, %arg88, %arg89) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %74 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%71#0, %22, %74) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %75 = call @BatchNormTrainingOp53(%74, %arg78, %arg79) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %76:2 = call @Unknown54(%75) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %77 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%76#0, %23, %77) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %78 = call @BatchNormTrainingOp55(%77, %arg83, %arg84) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %79:2 = call @Unknown56(%78, %73) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %80 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%79#0, %24, %80) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %81 = call @BatchNormTrainingOp57(%80, %arg93, %arg94) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %82:2 = call @Unknown58(%81) : (memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %83 = memref.alloc() : memref<4x512x7x7xf16>
    lmhlo.convolution(%82#0, %25, %83) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>) -> ()
    %84 = call @BatchNormTrainingOp59(%83, %arg98, %arg99) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>) -> memref<4x512x7x7xf16>
    %85:2 = call @Unknown60(%84, %79#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<4x512x7x7xi1>)
    %86 = memref.alloc() : memref<4x512xf16>
    "lmhlo.reduce"(%85#0, %1, %86) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (memref<4x512x7x7xf16>, memref<f16>, memref<4x512xf16>) -> ()
    %87 = call @Unknown61(%86) : (memref<4x512xf16>) -> memref<4x512xf16>
    %88 = memref.alloc() : memref<4x1000xf16>
    "lmhlo.dot"(%87, %27, %88) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>) -> ()
    %89 = call @Unknown62(%28, %88) : (memref<1000xf16>, memref<4x1000xf16>) -> memref<4x1000xf16>
    %90 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%89, %2, %90) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.maximum"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %91:2 = call @Unknown63(%90, %89) : (memref<4xf16>, memref<4x1000xf16>) -> (memref<4x1000xf16>, memref<4x1000xf16>)
    %92 = memref.alloc() : memref<4xf16>
    "lmhlo.reduce"(%91#1, %1, %92) ({
    ^bb0(%arg104: memref<f16>, %arg105: memref<f16>, %arg106: memref<f16>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f16>, memref<f16>, memref<f16>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<4x1000xf16>, memref<f16>, memref<4xf16>) -> ()
    %93 = call @Unknown64(%92) : (memref<4xf16>) -> memref<4xf16>
    %94:3 = call @Unknown65(%93, %91#0, %29, %26, %arg1) : (memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>) -> (memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>)
    %95 = memref.alloc() : memref<4x512xf16>
    "lmhlo.dot"(%94#0, %27, %95) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>) -> ()
    %96 = call @Unknown66(%95, %85#1) : (memref<4x512xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %97:3 = call @BatchNormGradOp67(%83, %arg98, %96) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %98 = call @ConvBackwardDataOp68(%97#0, %25) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %99 = call @ConvBackwardFilterOp69(%82#0, %97#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %100 = call @Unknown70(%82#1, %98) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %101:3 = call @BatchNormGradOp71(%80, %arg93, %100) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %102 = call @ConvBackwardDataOp72(%101#0, %24) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %103 = call @ConvBackwardFilterOp73(%79#0, %101#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %104 = call @Unknown74(%96, %102, %79#1) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>) -> memref<4x512x7x7xf16>
    %105:3 = call @BatchNormGradOp75(%77, %arg83, %104) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %106 = call @ConvBackwardDataOp76(%105#0, %23) : (memref<4x512x7x7xf16>, memref<512x512x3x3xf16>) -> memref<4x512x7x7xf16>
    %107 = call @ConvBackwardFilterOp77(%76#0, %105#0) : (memref<4x512x7x7xf16>, memref<4x512x7x7xf16>) -> memref<512x512x3x3xf16>
    %108 = call @Unknown78(%76#1, %106) : (memref<4x512x7x7xi1>, memref<4x512x7x7xf16>) -> memref<4x512x7x7xf16>
    %109:3 = call @BatchNormGradOp79(%74, %arg78, %108) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %110 = call @ConvBackwardDataOp80(%109#0, %22) : (memref<4x512x7x7xf16>, memref<512x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %111 = call @ConvBackwardFilterOp81(%71#0, %109#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x3x3xf16>
    %112:3 = call @BatchNormGradOp82(%72, %arg88, %104) : (memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>) -> (memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>)
    %113 = call @ConvBackwardDataOp83(%112#0, %21) : (memref<4x512x7x7xf16>, memref<512x256x1x1xf16>) -> memref<4x256x14x14xf16>
    %114 = call @ConvBackwardFilterOp84(%71#0, %112#0) : (memref<4x256x14x14xf16>, memref<4x512x7x7xf16>) -> memref<512x256x1x1xf16>
    %115 = call @Unknown85(%113, %110, %71#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %116:3 = call @BatchNormGradOp86(%69, %arg73, %115) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %117 = call @ConvBackwardDataOp87(%116#0, %20) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %118 = call @ConvBackwardFilterOp88(%68#0, %116#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %119 = call @Unknown89(%68#1, %117) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %120:3 = call @BatchNormGradOp90(%66, %arg68, %119) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %121 = call @ConvBackwardDataOp91(%120#0, %19) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %122 = call @ConvBackwardFilterOp92(%65#0, %120#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %123 = call @Unknown93(%115, %121, %65#1) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>) -> memref<4x256x14x14xf16>
    %124:3 = call @BatchNormGradOp94(%63, %arg58, %123) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %125 = call @ConvBackwardDataOp95(%124#0, %18) : (memref<4x256x14x14xf16>, memref<256x256x3x3xf16>) -> memref<4x256x14x14xf16>
    %126 = call @ConvBackwardFilterOp96(%62#0, %124#0) : (memref<4x256x14x14xf16>, memref<4x256x14x14xf16>) -> memref<256x256x3x3xf16>
    %127 = call @Unknown97(%62#1, %125) : (memref<4x256x14x14xi1>, memref<4x256x14x14xf16>) -> memref<4x256x14x14xf16>
    %128:3 = call @BatchNormGradOp98(%60, %arg53, %127) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %129 = call @ConvBackwardDataOp99(%128#0, %17) : (memref<4x256x14x14xf16>, memref<256x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %130 = call @ConvBackwardFilterOp100(%57#0, %128#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x3x3xf16>
    %131:3 = call @BatchNormGradOp101(%58, %arg63, %123) : (memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>) -> (memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>)
    %132 = call @ConvBackwardDataOp102(%131#0, %16) : (memref<4x256x14x14xf16>, memref<256x128x1x1xf16>) -> memref<4x128x28x28xf16>
    %133 = call @ConvBackwardFilterOp103(%57#0, %131#0) : (memref<4x128x28x28xf16>, memref<4x256x14x14xf16>) -> memref<256x128x1x1xf16>
    %134 = call @Unknown104(%132, %129, %57#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %135:3 = call @BatchNormGradOp105(%55, %arg48, %134) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %136 = call @ConvBackwardDataOp106(%135#0, %15) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %137 = call @ConvBackwardFilterOp107(%54#0, %135#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %138 = call @Unknown108(%54#1, %136) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %139:3 = call @BatchNormGradOp109(%52, %arg43, %138) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %140 = call @ConvBackwardDataOp110(%139#0, %14) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %141 = call @ConvBackwardFilterOp111(%51#0, %139#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %142 = call @Unknown112(%134, %140, %51#1) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>) -> memref<4x128x28x28xf16>
    %143:3 = call @BatchNormGradOp113(%49, %arg33, %142) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %144 = call @ConvBackwardDataOp114(%143#0, %13) : (memref<4x128x28x28xf16>, memref<128x128x3x3xf16>) -> memref<4x128x28x28xf16>
    %145 = call @ConvBackwardFilterOp115(%48#0, %143#0) : (memref<4x128x28x28xf16>, memref<4x128x28x28xf16>) -> memref<128x128x3x3xf16>
    %146 = call @Unknown116(%48#1, %144) : (memref<4x128x28x28xi1>, memref<4x128x28x28xf16>) -> memref<4x128x28x28xf16>
    %147:3 = call @BatchNormGradOp117(%46, %arg28, %146) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %148 = call @ConvBackwardDataOp118(%147#0, %12) : (memref<4x128x28x28xf16>, memref<128x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %149 = call @ConvBackwardFilterOp119(%43#0, %147#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x3x3xf16>
    %150:3 = call @BatchNormGradOp120(%44, %arg38, %142) : (memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>) -> (memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>)
    %151 = call @ConvBackwardDataOp121(%150#0, %11) : (memref<4x128x28x28xf16>, memref<128x64x1x1xf16>) -> memref<4x64x56x56xf16>
    %152 = call @ConvBackwardFilterOp122(%43#0, %150#0) : (memref<4x64x56x56xf16>, memref<4x128x28x28xf16>) -> memref<128x64x1x1xf16>
    %153 = call @Unknown123(%151, %148, %43#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %154:3 = call @BatchNormGradOp124(%41, %arg23, %153) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %155 = call @ConvBackwardDataOp125(%154#0, %10) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %156 = call @ConvBackwardFilterOp126(%40#0, %154#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %157 = call @Unknown127(%40#1, %155) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %158:3 = call @BatchNormGradOp128(%38, %arg18, %157) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %159 = call @ConvBackwardDataOp129(%158#0, %9) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %160 = call @ConvBackwardFilterOp130(%37#0, %158#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %161 = call @Unknown131(%153, %159, %37#1) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>) -> memref<4x64x56x56xf16>
    %162:3 = call @BatchNormGradOp132(%35, %arg13, %161) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %163 = call @ConvBackwardDataOp133(%162#0, %8) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %164 = call @ConvBackwardFilterOp134(%34#0, %162#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %165 = call @Unknown135(%34#1, %163) : (memref<4x64x56x56xi1>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %166:3 = call @BatchNormGradOp136(%32, %arg8, %165) : (memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>) -> (memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>)
    %167 = call @ConvBackwardDataOp137(%166#0, %7) : (memref<4x64x56x56xf16>, memref<64x64x3x3xf16>) -> memref<4x64x56x56xf16>
    %168 = call @ConvBackwardFilterOp138(%31, %166#0) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<64x64x3x3xf16>
    %169 = call @Unknown139(%161, %167) : (memref<4x64x56x56xf16>, memref<4x64x56x56xf16>) -> memref<4x64x56x56xf16>
    %170 = memref.alloc() : memref<4x64x112x112xf16>
    "lmhlo.select_and_scatter"(%30#0, %169, %1, %170) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %200 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%200) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %200 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%200) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<f16>, memref<4x64x112x112xf16>) -> ()
    %171 = call @Unknown140(%30#1, %170) : (memref<4x64x112x112xi1>, memref<4x64x112x112xf16>) -> memref<4x64x112x112xf16>
    %172:3 = call @BatchNormGradOp141(%5, %arg3, %171) : (memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>) -> (memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>)
    %173 = call @ConvBackwardFilterOp142(%3, %172#0) : (memref<4x3x224x224xf16>, memref<4x64x112x112xf16>) -> memref<64x3x7x7xf16>
    %174 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%94#1, %0, %174) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<4x1000xf32>, memref<f32>, memref<f32>) -> ()
    %175 = call @Unknown143(%174) : (memref<f32>) -> memref<f32>
    %176 = call @Unknown144(%173) : (memref<64x3x7x7xf16>) -> memref<64x3x7x7xf32>
    %177 = call @Unknown145(%168) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %178 = call @Unknown146(%164) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %179 = call @Unknown147(%160) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %180 = call @Unknown148(%156) : (memref<64x64x3x3xf16>) -> memref<64x64x3x3xf32>
    %181 = call @Unknown149(%149) : (memref<128x64x3x3xf16>) -> memref<128x64x3x3xf32>
    %182 = call @Unknown150(%145) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %183 = call @Unknown151(%152) : (memref<128x64x1x1xf16>) -> memref<128x64x1x1xf32>
    %184 = call @Unknown152(%141) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %185 = call @Unknown153(%137) : (memref<128x128x3x3xf16>) -> memref<128x128x3x3xf32>
    %186 = call @Unknown154(%130) : (memref<256x128x3x3xf16>) -> memref<256x128x3x3xf32>
    %187 = call @Unknown155(%126) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %188 = call @Unknown156(%133) : (memref<256x128x1x1xf16>) -> memref<256x128x1x1xf32>
    %189 = call @Unknown157(%122) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %190 = call @Unknown158(%118) : (memref<256x256x3x3xf16>) -> memref<256x256x3x3xf32>
    %191 = call @Unknown159(%111) : (memref<512x256x3x3xf16>) -> memref<512x256x3x3xf32>
    %192 = call @Unknown160(%107) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %193 = call @Unknown161(%114) : (memref<512x256x1x1xf16>) -> memref<512x256x1x1xf32>
    %194 = call @Unknown162(%103) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %195 = call @Unknown163(%99) : (memref<512x512x3x3xf16>) -> memref<512x512x3x3xf32>
    %196 = call @MatmulOp164(%87, %94#0) : (memref<4x512xf16>, memref<4x1000xf16>) -> memref<1000x512xf16>
    %197 = call @Unknown165(%196) : (memref<1000x512xf16>) -> memref<1000x512xf32>
    %198 = memref.alloc() : memref<1000xf32>
    "lmhlo.reduce"(%94#2, %0, %198) ({
    ^bb0(%arg104: memref<f32>, %arg105: memref<f32>, %arg106: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg104, %arg105, %arg106) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<4x1000xf32>, memref<f32>, memref<1000xf32>) -> ()
    %199 = call @Unknown166(%198) : (memref<1000xf32>) -> memref<1000xf32>
    return %175, %176, %172#1, %172#2, %177, %166#1, %166#2, %178, %162#1, %162#2, %179, %158#1, %158#2, %180, %154#1, %154#2, %181, %147#1, %147#2, %182, %143#1, %143#2, %183, %150#1, %150#2, %184, %139#1, %139#2, %185, %135#1, %135#2, %186, %128#1, %128#2, %187, %124#1, %124#2, %188, %131#1, %131#2, %189, %120#1, %120#2, %190, %116#1, %116#2, %191, %109#1, %109#2, %192, %105#1, %105#2, %193, %112#1, %112#2, %194, %101#1, %101#2, %195, %97#1, %97#2, %197, %199 : memref<f32>, memref<64x3x7x7xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<64x64x3x3xf32>, memref<64xf32>, memref<64xf32>, memref<128x64x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x64x1x1xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<128x128x3x3xf32>, memref<128xf32>, memref<128xf32>, memref<256x128x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x128x1x1xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<256x256x3x3xf32>, memref<256xf32>, memref<256xf32>, memref<512x256x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x256x1x1xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<512x512x3x3xf32>, memref<512xf32>, memref<512xf32>, memref<1000x512xf32>, memref<1000xf32>
  }
}

