// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0 * 128 + d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<() -> ()>
#map10 = affine_map<(d0, d1) -> ()>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-100_i64 = arith.constant -100 : i64
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    %1 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%0 : memref<256xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):
      %2 = arith.cmpi ne, %arg1, %c-100_i64 : i64
      linalg.yield %2 : i1
    }
    return %1, %0 : memref<256xi64>, memref<256xi1>
  }
  func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c30522_i64 = arith.constant 30522 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xui32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256xi64>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%0 : memref<256xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):
      %4 = arith.trunci %arg1 : i64 to i32
      %5 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      linalg.yield %5 : ui32
    }
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%1 : memref<256xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %4 = arith.addi %arg1, %c30522_i64 : i64
      %5 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %6 = arith.select %5, %4, %arg1 : i64
      linalg.yield %6 : i64
    }
    %3 = memref.expand_shape %1 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%2 : memref<256xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):
      %4 = arith.cmpi ne, %arg1, %c0_i64 : i64
      linalg.yield %4 : i1
    }
    return %0, %3, %2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128xui32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<2x128xi64>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%0 : memref<2x128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):
      %7 = arith.trunci %arg1 : i64 to i32
      %8 = builtin.unrealized_conversion_cast %7 : i32 to ui32
      linalg.yield %8 : ui32
    }
    %3 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%1 : memref<2x128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %7 = arith.addi %arg1, %c2_i64 : i64
      %8 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %9 = arith.select %8, %7, %arg1 : i64
      linalg.yield %9 : i64
    }
    %4 = memref.collapse_shape %1 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %5 = memref.expand_shape %4 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%2 : memref<2x128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):
      %7 = arith.cmpi ne, %arg1, %c-1_i64 : i64
      linalg.yield %7 : i1
    }
    %6 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %3, %5, %6 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-1_i64 = arith.constant -1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xui32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<128xi64>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<128xi1>
    %3 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%3 : memref<128xi64>) outs(%0 : memref<128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):
      %5 = arith.trunci %arg1 : i64 to i32
      %6 = builtin.unrealized_conversion_cast %5 : i32 to ui32
      linalg.yield %6 : ui32
    }
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%3 : memref<128xi64>) outs(%1 : memref<128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %5 = arith.addi %arg1, %c512_i64 : i64
      %6 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %7 = arith.select %6, %5, %arg1 : i64
      linalg.yield %7 : i64
    }
    %4 = memref.expand_shape %1 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%3 : memref<128xi64>) outs(%2 : memref<128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):
      %5 = arith.cmpi ne, %arg1, %c-1_i64 : i64
      linalg.yield %5 : i1
    }
    return %0, %4, %2 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %1 = arith.addf %arg3, %arg4 : f32
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %2 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    linalg.generic {indexing_maps = [#map4, #map7, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<256x30522xf32>, memref<30522xf32>) outs(%0 : memref<2x128x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %3 = arith.addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map6, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %arg1 : memref<2x128x30522xf32>, memref<30522xf32>) outs(%1 : memref<256x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %3 = arith.addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    }
    return %0, %1 : memref<2x128x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map0, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<256xf32>) outs(%0 : memref<256x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %2 = arith.subf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    }
    linalg.generic {indexing_maps = [#map0, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<256xf32>) outs(%1 : memref<256x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %2 = arith.subf %arg2, %arg3 : f32
      %3 = math.exp %2 : f32
      linalg.yield %3 : f32
    }
    return %0, %1 : memref<256x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%arg0 : memref<256xf32>) outs(%0 : memref<256xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = math.log %arg1 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256xf32>
  }
  func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map8, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2 : memref<256xi1>, memref<256xi64>) outs(%0 : memref<256x30522xf32>) {
    ^bb0(%arg4: i1, %arg5: i64, %arg6: f32):
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.cmpi eq, %arg5, %5 : i64
      %7 = arith.select %6, %cst, %cst_0 : f32
      %8 = arith.select %arg4, %cst, %cst_0 : f32
      %9 = arith.mulf %8, %7 : f32
      linalg.yield %9 : f32
    }
    linalg.generic {indexing_maps = [#map8, #map0, #map8, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg1, %arg0, %arg3 : memref<256xi64>, memref<256x30522xf32>, memref<256xf32>, memref<256xi1>) outs(%1 : memref<256x30522xf32>) {
    ^bb0(%arg4: i64, %arg5: f32, %arg6: f32, %arg7: i1, %arg8: f32):
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.cmpi eq, %arg4, %5 : i64
      %7 = arith.select %6, %cst, %cst_0 : f32
      %8 = arith.select %arg7, %cst, %cst_0 : f32
      %9 = arith.mulf %8, %7 : f32
      %10 = arith.subf %arg5, %arg6 : f32
      %11 = arith.negf %7 : f32
      %12 = arith.mulf %11, %10 : f32
      %13 = arith.cmpf une, %7, %cst : f32
      %14 = arith.select %13, %cst_0, %12 : f32
      %15 = arith.mulf %14, %9 : f32
      linalg.yield %15 : f32
    }
    linalg.generic {indexing_maps = [#map8, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg3 : memref<256xi64>, memref<256xi1>) outs(%2 : memref<256x30522xf32>) {
    ^bb0(%arg4: i64, %arg5: i1, %arg6: f32):
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.cmpi eq, %arg4, %5 : i64
      %7 = arith.select %6, %cst, %cst_0 : f32
      %8 = arith.select %arg5, %cst, %cst_0 : f32
      %9 = arith.mulf %8, %7 : f32
      %10 = arith.negf %7 : f32
      %11 = arith.mulf %10, %9 : f32
      linalg.yield %11 : f32
    }
    linalg.generic {indexing_maps = [#map0, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<256xf32>) outs(%3 : memref<256x30522xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %4 = arith.subf %arg4, %arg5 : f32
      %5 = math.exp %4 : f32
      linalg.yield %5 : f32
    }
    return %0, %1, %2, %3 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
  }
  func private @Unknown9(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map9, #map9], iterator_types = []} ins(%arg0 : memref<f32>) outs(%0 : memref<f32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.cmpf une, %arg1, %cst_0 : f32
      %2 = arith.select %1, %arg1, %cst : f32
      linalg.yield %2 : f32
    }
    return %0 : memref<f32>
  }
  func private @Unknown10(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map0, #map10, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<f32>) outs(%0 : memref<256x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.divf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<256x30522xf32>
  }
  func private @Unknown11(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = []} ins(%arg0, %arg1 : memref<f32>, memref<f32>) outs(%0 : memref<f32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.divf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<f32>
  }
  func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<2x128x30522xf32>
    linalg.generic {indexing_maps = [#map0, #map0, #map8, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg2, %arg1, %arg0 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256xf32>) outs(%0 : memref<256x30522xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.mulf %arg4, %arg5 : f32
      %4 = arith.subf %arg3, %3 : f32
      linalg.yield %4 : f32
    }
    %2 = memref.expand_shape %arg0 [[0, 1]] : memref<256xf32> into memref<2x128xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map11, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2, %arg1, %2 : memref<256x30522xf32>, memref<256x30522xf32>, memref<2x128xf32>) outs(%1 : memref<2x128x30522xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):
      %3 = arith.mulf %arg4, %arg5 : f32
      %4 = arith.subf %arg3, %3 : f32
      linalg.yield %4 : f32
    }
    return %0, %1 : memref<256x30522xf32>, memref<2x128x30522xf32>
  }
  func private @MatmulOp13(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
      %1 = arith.addf %arg4, %arg5 : f32
      %2 = arith.addf %1, %arg6 : f32
      %3 = arith.addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %1 = arith.addf %arg2, %arg3 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
      %1 = arith.addf %arg4, %arg5 : f32
      %2 = arith.addf %1, %arg6 : f32
      %3 = arith.addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    %2 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    linalg.generic {indexing_maps = [#map11, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %arg1 : memref<2x128xi1>, memref<2x128x128xf32>) outs(%0 : memref<256x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):
      %4 = arith.select %arg3, %arg4, %cst : f32
      linalg.yield %4 : f32
    }
    %3 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    linalg.generic {indexing_maps = [#map11, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %arg1 : memref<2x128xi1>, memref<2x128x128xf32>) outs(%1 : memref<256x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):
      %4 = arith.select %arg3, %arg4, %cst : f32
      linalg.yield %4 : f32
    }
    return %0, %1 : memref<256x128xf32>, memref<256x128xf32>
  }
  func private @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map8, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<128xi1>, memref<128x128xf32>) outs(%0 : memref<128x128xf32>) {
    ^bb0(%arg2: i1, %arg3: f32, %arg4: f32):
      %1 = arith.select %arg2, %arg3, %cst : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128xf32>
  }
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %2 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%2) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %3 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%3) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    %4 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.constant"(%4) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    %5 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %5) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %6 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%5, %6) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %7 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg3, %7) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %8:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %9:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %10 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %9#0, %10) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %11:3 = call @Unknown2(%6) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %12 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg5, %11#0, %12) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %13:3 = call @Unknown3(%7) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %14 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg6, %13#0, %14) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %15 = call @Unknown4(%10, %12, %14) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    %16 = memref.alloc() : memref<2x128x128xf32>
    %17 = memref.alloc() : memref<256xf32>
    %18 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%15, %arg7, %arg8, %16, %17, %18) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %19 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg9, %arg10, %19) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %20 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg11, %arg12, %20) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %21 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%19, %20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %22 = memref.alloc() : memref<2x2x128x128xf32>
    %23 = memref.alloc() : memref<2x2x128x128xf32>
    %24 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%21, %4, %22, %23, %24) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg13, %arg14, %25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %26 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%22, %25, %26) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %27 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%26, %27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %28 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%27, %28) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %29 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%28, %arg15, %arg16, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %30 = memref.alloc() : memref<2x128x128xf32>
    %31 = memref.alloc() : memref<256xf32>
    %32 = memref.alloc() : memref<256xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%29, %arg17, %arg18, %16, %30, %31, %32, %33) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %34 = memref.alloc() : memref<2x128x512xf32>
    %35 = memref.alloc() : memref<2x128x512xf32>
    %36 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%30, %arg19, %arg20, %34, %35, %36) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %37 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%34, %arg21, %arg22, %37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<256xf32>
    %40 = memref.alloc() : memref<256xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%37, %arg23, %arg24, %30, %38, %39, %40, %41) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %42 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg25, %arg26, %42) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %43 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg27, %arg28, %43) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %44 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%42, %43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %45 = memref.alloc() : memref<2x2x128x128xf32>
    %46 = memref.alloc() : memref<2x2x128x128xf32>
    %47 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%44, %4, %45, %46, %47) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %48 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg29, %arg30, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %49 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%45, %48, %49) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %50 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%49, %50) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %51 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%50, %51) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %52 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%51, %arg31, %arg32, %52) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %53 = memref.alloc() : memref<2x128x128xf32>
    %54 = memref.alloc() : memref<256xf32>
    %55 = memref.alloc() : memref<256xf32>
    %56 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%52, %arg33, %arg34, %38, %53, %54, %55, %56) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %57 = memref.alloc() : memref<2x128x512xf32>
    %58 = memref.alloc() : memref<2x128x512xf32>
    %59 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%53, %arg35, %arg36, %57, %58, %59) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %60 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%57, %arg37, %arg38, %60) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<256xf32>
    %63 = memref.alloc() : memref<256xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%60, %arg39, %arg40, %53, %61, %62, %63, %64) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %65 = memref.alloc() : memref<2x128x128xf32>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%61, %arg41, %arg42, %65, %66, %67) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %68 = memref.alloc() : memref<2x128x128xf32>
    %69 = memref.alloc() : memref<256xf32>
    %70 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%65, %arg43, %arg44, %68, %69, %70) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %71 = memref.alloc() : memref<256x128xf32>
    "lmhlo.reshape"(%68, %71) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    %72 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.dot"(%71, %arg4, %72) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %73:2 = call @Unknown5(%72, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    %74 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%73#1, %3, %74) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %75:2 = call @Unknown6(%74, %73#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    %76 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%75#1, %0, %76) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %77 = call @Unknown7(%76) : (memref<256xf32>) -> memref<256xf32>
    %78:4 = call @Unknown8(%77, %75#0, %8#0, %8#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    %79 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#0, %0, %79) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %80 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#0, %0, %80) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %81 = call @Unknown9(%80) : (memref<f32>) -> memref<f32>
    %82 = call @Unknown10(%81, %78#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    %83 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%82, %0, %83) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %84 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#1, %0, %84) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %85 = call @Unknown11(%84, %79) : (memref<f32>, memref<f32>) -> memref<f32>
    %86:2 = call @Unknown12(%83, %78#3, %82) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %87 = call @MatmulOp13(%71, %86#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    %88 = memref.alloc() : memref<256x128xf32>
    "lmhlo.dot"(%86#0, %arg4, %88) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    %89 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%88, %89) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<128xf32>
    %92 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%89, %65, %arg43, %69, %70, %90, %91, %92) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<128x128xf32>
    %95 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%90, %61, %arg41, %66, %67, %93, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<128xf32>
    %98 = memref.alloc() : memref<128xf32>
    %99 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%93, %64, %arg39, %62, %63, %96, %97, %98, %99) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %100 = memref.alloc() : memref<2x128x512xf32>
    %101 = memref.alloc() : memref<128x512xf32>
    %102 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%96, %57, %arg37, %100, %101, %102) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %103 = memref.alloc() : memref<2x128x128xf32>
    %104 = memref.alloc() : memref<512x128xf32>
    %105 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%100, %53, %arg35, %58, %59, %103, %104, %105) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %106 = call @Unknown14(%99, %103) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %107 = memref.alloc() : memref<2x128x128xf32>
    %108 = memref.alloc() : memref<128xf32>
    %109 = memref.alloc() : memref<128xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%106, %56, %arg33, %54, %55, %107, %108, %109, %110) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %111 = memref.alloc() : memref<2x128x128xf32>
    %112 = memref.alloc() : memref<128x128xf32>
    %113 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%107, %51, %arg31, %111, %112, %113) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %114 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%111, %114) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %115 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%114, %115) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%115, %45, %48, %116, %117) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %118 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%116, %45, %47, %118) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %119 = memref.alloc() : memref<2x2x128x64xf32>
    %120 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%118, %42, %43, %119, %120) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %121 = memref.alloc() : memref<2x128x128xf32>
    %122 = memref.alloc() : memref<128x128xf32>
    %123 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%119, %38, %arg25, %121, %122, %123) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<128x128xf32>
    %126 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%117, %38, %arg29, %124, %125, %126) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %127 = memref.alloc() : memref<2x128x128xf32>
    %128 = memref.alloc() : memref<128x128xf32>
    %129 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%120, %38, %arg27, %127, %128, %129) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %130 = call @Unknown15(%110, %121, %124, %127) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    %132 = memref.alloc() : memref<128xf32>
    %133 = memref.alloc() : memref<128xf32>
    %134 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%130, %41, %arg23, %39, %40, %131, %132, %133, %134) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %135 = memref.alloc() : memref<2x128x512xf32>
    %136 = memref.alloc() : memref<128x512xf32>
    %137 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%131, %34, %arg21, %135, %136, %137) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %138 = memref.alloc() : memref<2x128x128xf32>
    %139 = memref.alloc() : memref<512x128xf32>
    %140 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%135, %30, %arg19, %35, %36, %138, %139, %140) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %141 = call @Unknown16(%134, %138) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %142 = memref.alloc() : memref<2x128x128xf32>
    %143 = memref.alloc() : memref<128xf32>
    %144 = memref.alloc() : memref<128xf32>
    %145 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%141, %33, %arg17, %31, %32, %142, %143, %144, %145) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %146 = memref.alloc() : memref<2x128x128xf32>
    %147 = memref.alloc() : memref<128x128xf32>
    %148 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%142, %28, %arg15, %146, %147, %148) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %149 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%146, %149) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %150 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%149, %150) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %151 = memref.alloc() : memref<2x2x128x128xf32>
    %152 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%150, %22, %25, %151, %152) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %153 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%151, %22, %24, %153) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %154 = memref.alloc() : memref<2x2x128x64xf32>
    %155 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%153, %19, %20, %154, %155) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %156 = memref.alloc() : memref<2x128x128xf32>
    %157 = memref.alloc() : memref<128x128xf32>
    %158 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%154, %16, %arg9, %156, %157, %158) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %159 = memref.alloc() : memref<2x128x128xf32>
    %160 = memref.alloc() : memref<128x128xf32>
    %161 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%152, %16, %arg13, %159, %160, %161) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %162 = memref.alloc() : memref<2x128x128xf32>
    %163 = memref.alloc() : memref<128x128xf32>
    %164 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%155, %16, %arg11, %162, %163, %164) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %165 = call @Unknown17(%145, %156, %159, %162) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %166 = memref.alloc() : memref<2x128x128xf32>
    %167 = memref.alloc() : memref<128xf32>
    %168 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%165, %15, %arg7, %17, %18, %166, %167, %168) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %169:2 = call @Unknown18(%9#2, %166, %11#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %170 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%87, %9#1, %169#0, %170) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %171 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%2, %11#1, %169#1, %171) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %172 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%166, %0, %172) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %173 = call @Unknown19(%13#2, %172) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    %174 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%1, %13#1, %173, %174) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    %175 = memref.alloc() : memref<30522xf32>
    "lmhlo.reduce"(%86#1, %0, %175) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %73#0, %85, %170, %171, %174, %167, %168, %157, %158, %163, %164, %160, %161, %147, %148, %143, %144, %139, %140, %136, %137, %132, %133, %122, %123, %128, %129, %125, %126, %112, %113, %108, %109, %104, %105, %101, %102, %97, %98, %94, %95, %91, %92, %175 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

