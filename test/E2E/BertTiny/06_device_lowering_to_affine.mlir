// RUN: byteir-opt %s -convert-linalg-to-affine-loops -loop-coalescing -simplify-affine-structures -affine-loop-fusion -affine-loop-fusion-ex -cse -cmae | FileCheck %s

// CHECK-LABEL: func @main
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0)>
module  {
  func private @MatmulOp0(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c0_i64 = arith.constant 0 : i64
    %c30522_i64 = arith.constant 30522 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %0 = memref.alloc() : memref<2x128xui32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%0 : memref<2x128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %7 = arith.trunci %arg1 : i64 to i32
      %8 = builtin.unrealized_conversion_cast %7 : i32 to ui32
      linalg.yield %8 : ui32
    }
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %2 = memref.alloc() : memref<2x128xi64>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%2 : memref<2x128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %7 = arith.addi %arg1, %c30522_i64 : i64
      %8 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %9 = select %8, %7, %arg1 : i64
      linalg.yield %9 : i64
    }
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%5 : memref<2x128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %7 = arith.sitofp %arg1 : i64 to f64
      %8 = arith.cmpf une, %7, %cst : f64
      linalg.yield %8 : i1
    }
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %1, %4, %6 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown1(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant -1.000000e+00 : f64
    %0 = memref.alloc() : memref<2x128xui32>
    linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%0 : memref<2x128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %7 = arith.trunci %arg1 : i64 to i32
      %8 = builtin.unrealized_conversion_cast %7 : i32 to ui32
      linalg.yield %8 : ui32
    }
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %2 = memref.alloc() : memref<2x128xi64>
    linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%2 : memref<2x128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %7 = arith.addi %arg1, %c2_i64 : i64
      %8 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %9 = select %8, %7, %arg1 : i64
      linalg.yield %9 : i64
    }
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%5 : memref<2x128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %7 = arith.sitofp %arg1 : i64 to f64
      %8 = arith.cmpf une, %7, %cst : f64
      linalg.yield %8 : i1
    }
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %1, %4, %6 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown2(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %1 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%2 : memref<2x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = arith.addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    }
    return %2 : memref<2x128x128xf32>
  }
  func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_elementwise_fusion} {
    %c0_i64 = arith.constant 0 : i64
    %c512_i64 = arith.constant 512 : i64
    %cst = arith.constant -1.000000e+00 : f64
    %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %1 = memref.alloc() : memref<128xui32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : memref<128xi64>) outs(%1 : memref<128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %5 = arith.trunci %arg1 : i64 to i32
      %6 = builtin.unrealized_conversion_cast %5 : i32 to ui32
      linalg.yield %6 : ui32
    }
    %2 = memref.alloc() : memref<128xi64>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : memref<128xi64>) outs(%2 : memref<128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %5 = arith.addi %arg1, %c512_i64 : i64
      %6 = arith.cmpi slt, %arg1, %c0_i64 : i64
      %7 = select %6, %5, %arg1 : i64
      linalg.yield %7 : i64
    }
    %3 = memref.expand_shape %2 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %4 = memref.alloc() : memref<128xi1>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : memref<128xi64>) outs(%4 : memref<128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %5 = arith.sitofp %arg1 : i64 to f64
      %6 = arith.cmpf une, %5, %cst : f64
      linalg.yield %6 : i1
    }
    return %1, %3, %4 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func private @Unknown4(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> memref<2x128x30522xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %1 = memref.alloc() : memref<2x128x30522xf32>
    linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : memref<2x128x30522xf32>, memref<30522xf32>) outs(%1 : memref<2x128x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = arith.addf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    }
    return %1 : memref<2x128x30522xf32>
  }
  func private @Unknown5(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f32
      %2 = arith.addf %1, %arg6 : f32
      %3 = arith.addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown6(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %1 = arith.addf %arg4, %arg5 : f32
      %2 = arith.addf %1, %arg6 : f32
      %3 = arith.addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown7(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {byre_elementwise_fusion} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %1 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map5, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : memref<2x128xi1>, memref<2x128x128xf32>) outs(%1 : memref<2x128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):  // no predecessors
      %6 = select %arg3, %arg4, %cst : f32
      linalg.yield %6 : f32
    }
    %2 = memref.collapse_shape %1 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %3 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %4 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map5, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %arg1 : memref<2x128xi1>, memref<2x128x128xf32>) outs(%4 : memref<2x128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):  // no predecessors
      %6 = select %arg3, %arg4, %cst : f32
      linalg.yield %6 : f32
    }
    %5 = memref.collapse_shape %4 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    return %2, %5 : memref<256x128xf32>, memref<256x128xf32>
  }
  func private @Unknown8(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {byre_elementwise_fusion} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map6, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<128xi1>, memref<128x128xf32>) outs(%0 : memref<128x128xf32>) {
    ^bb0(%arg2: i1, %arg3: f32, %arg4: f32):  // no predecessors
      %1 = select %arg2, %arg3, %cst : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128xf32>
  }
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<1x512xi64>, %arg2: memref<1x512xi64>, %arg3: memref<30522x128xf32>, %arg4: memref<2x128xf32>, %arg5: memref<512x128xf32>, %arg6: memref<128xf32>, %arg7: memref<128xf32>, %arg8: memref<128x128xf32>, %arg9: memref<128xf32>, %arg10: memref<128x128xf32>, %arg11: memref<128xf32>, %arg12: memref<128x128xf32>, %arg13: memref<128xf32>, %arg14: memref<2x1x1x128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<2x1x1x128xf32>, %arg32: memref<128x128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<512x128xf32>, %arg37: memref<512xf32>, %arg38: memref<128x512xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<30522xf32>, %arg47: memref<2x128x30522xf32>) -> (memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %1 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %2 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%2) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %3 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg1, %3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %4 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%3, %4) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %5 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %5) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %6 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.reshape"(%arg47, %6) : (memref<2x128x30522xf32>, memref<256x30522xf32>) -> ()
    %7:3 = call @Unknown0(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %8 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg3, %7#0, %8) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %9 = memref.alloc() : memref<256x128xf32>
    "lmhlo.dot"(%6, %arg3, %9) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    %10 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%9, %10) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %11:3 = call @Unknown1(%4) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %12 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %11#0, %12) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %13 = call @Unknown2(%8, %12) : (memref<256x128xf32>, memref<256x128xf32>) -> memref<2x128x128xf32>
    %14:3 = call @Unknown3(%5) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %15 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg5, %14#0, %15) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %16 = memref.alloc() : memref<1x128x128xf32>
    "lmhlo.reshape"(%15, %16) : (memref<128x128xf32>, memref<1x128x128xf32>) -> ()
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<256xf32>
    %19 = memref.alloc() : memref<256xf32>
    %20 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%13, %arg6, %arg7, %16, %17, %18, %19, %20) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %21 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%17, %arg8, %arg9, %21) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %22 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%17, %arg10, %arg11, %22) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %23 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%21, %22, %23) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %24 = memref.alloc() : memref<2x2x128x128xf32>
    %25 = memref.alloc() : memref<2x2x128x128xf32>
    %26 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%23, %arg14, %24, %25, %26) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %27 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%17, %arg12, %arg13, %27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %28 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%25, %27, %28) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %29 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%28, %29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %30 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%29, %30) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %31 = memref.alloc() : memref<2x128x128xf32>
    %32 = memref.alloc() : memref<2x128x128xf32>
    %33 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%30, %arg15, %arg16, %31, %32, %33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<256xf32>
    %36 = memref.alloc() : memref<256xf32>
    %37 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%31, %arg17, %arg18, %17, %34, %35, %36, %37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %38 = memref.alloc() : memref<2x128x512xf32>
    %39 = memref.alloc() : memref<2x128x512xf32>
    %40 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%34, %arg19, %arg20, %38, %39, %40) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<2x128x128xf32>
    %43 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%38, %arg21, %arg22, %41, %42, %43) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %44 = memref.alloc() : memref<2x128x128xf32>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<256xf32>
    %47 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%41, %arg23, %arg24, %34, %44, %45, %46, %47) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %48 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%44, %arg25, %arg26, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %49 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%44, %arg27, %arg28, %49) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %50 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%48, %49, %50) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %51 = memref.alloc() : memref<2x2x128x128xf32>
    %52 = memref.alloc() : memref<2x2x128x128xf32>
    %53 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%50, %arg31, %51, %52, %53) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%44, %arg29, %arg30, %54) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %55 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%52, %54, %55) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %56 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%55, %56) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %57 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%56, %57) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %58 = memref.alloc() : memref<2x128x128xf32>
    %59 = memref.alloc() : memref<2x128x128xf32>
    %60 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%57, %arg32, %arg33, %58, %59, %60) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<256xf32>
    %63 = memref.alloc() : memref<256xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%58, %arg34, %arg35, %44, %61, %62, %63, %64) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %65 = memref.alloc() : memref<2x128x512xf32>
    %66 = memref.alloc() : memref<2x128x512xf32>
    %67 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%61, %arg36, %arg37, %65, %66, %67) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %68 = memref.alloc() : memref<2x128x128xf32>
    %69 = memref.alloc() : memref<2x128x128xf32>
    %70 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%65, %arg38, %arg39, %68, %69, %70) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %71 = memref.alloc() : memref<2x128x128xf32>
    %72 = memref.alloc() : memref<256xf32>
    %73 = memref.alloc() : memref<256xf32>
    %74 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%68, %arg40, %arg41, %61, %71, %72, %73, %74) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %75 = memref.alloc() : memref<2x128x128xf32>
    %76 = memref.alloc() : memref<2x128x128xf32>
    %77 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%71, %arg42, %arg43, %75, %76, %77) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %78 = memref.alloc() : memref<2x128x128xf32>
    %79 = memref.alloc() : memref<256xf32>
    %80 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%75, %arg44, %arg45, %78, %79, %80) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %81 = memref.alloc() : memref<256x128xf32>
    "lmhlo.reshape"(%78, %81) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    %82 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.dot"(%81, %arg3, %82) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %83 = call @Unknown4(%82, %arg46) : (memref<256x30522xf32>, memref<30522xf32>) -> memref<2x128x30522xf32>
    %84 = call @MatmulOp0(%81, %6) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    %85 = memref.alloc() : memref<2x128x128xf32>
    %86 = memref.alloc() : memref<128xf32>
    %87 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%10, %75, %arg44, %79, %80, %85, %86, %87) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %88 = memref.alloc() : memref<2x128x128xf32>
    %89 = memref.alloc() : memref<128x128xf32>
    %90 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%85, %71, %arg42, %76, %77, %88, %89, %90) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %91 = memref.alloc() : memref<2x128x128xf32>
    %92 = memref.alloc() : memref<128xf32>
    %93 = memref.alloc() : memref<128xf32>
    %94 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%88, %74, %arg40, %72, %73, %91, %92, %93, %94) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %95 = memref.alloc() : memref<2x128x512xf32>
    %96 = memref.alloc() : memref<128x512xf32>
    %97 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%91, %65, %arg38, %69, %70, %95, %96, %97) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %98 = memref.alloc() : memref<2x128x128xf32>
    %99 = memref.alloc() : memref<512x128xf32>
    %100 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%95, %61, %arg36, %66, %67, %98, %99, %100) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %101 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.add"(%94, %98, %101) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    %102 = memref.alloc() : memref<2x128x128xf32>
    %103 = memref.alloc() : memref<128xf32>
    %104 = memref.alloc() : memref<128xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%101, %64, %arg34, %62, %63, %102, %103, %104, %105) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %106 = memref.alloc() : memref<2x128x128xf32>
    %107 = memref.alloc() : memref<128x128xf32>
    %108 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%102, %57, %arg32, %59, %60, %106, %107, %108) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %109 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%106, %109) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %110 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%109, %110) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %111 = memref.alloc() : memref<2x2x128x128xf32>
    %112 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%110, %52, %54, %111, %112) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %113 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%111, %51, %53, %113) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %114 = memref.alloc() : memref<2x2x128x64xf32>
    %115 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%113, %48, %49, %114, %115) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %116 = memref.alloc() : memref<2x128x128xf32>
    %117 = memref.alloc() : memref<128x128xf32>
    %118 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%114, %44, %arg25, %116, %117, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %119 = memref.alloc() : memref<2x128x128xf32>
    %120 = memref.alloc() : memref<128x128xf32>
    %121 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%112, %44, %arg29, %119, %120, %121) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %122 = memref.alloc() : memref<2x128x128xf32>
    %123 = memref.alloc() : memref<128x128xf32>
    %124 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%115, %44, %arg27, %122, %123, %124) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %125 = call @Unknown5(%105, %116, %119, %122) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %126 = memref.alloc() : memref<2x128x128xf32>
    %127 = memref.alloc() : memref<128xf32>
    %128 = memref.alloc() : memref<128xf32>
    %129 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%125, %47, %arg23, %45, %46, %126, %127, %128, %129) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %130 = memref.alloc() : memref<2x128x512xf32>
    %131 = memref.alloc() : memref<128x512xf32>
    %132 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%126, %38, %arg21, %42, %43, %130, %131, %132) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %133 = memref.alloc() : memref<2x128x128xf32>
    %134 = memref.alloc() : memref<512x128xf32>
    %135 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%130, %34, %arg19, %39, %40, %133, %134, %135) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %136 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.add"(%129, %133, %136) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    %137 = memref.alloc() : memref<2x128x128xf32>
    %138 = memref.alloc() : memref<128xf32>
    %139 = memref.alloc() : memref<128xf32>
    %140 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%136, %37, %arg17, %35, %36, %137, %138, %139, %140) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %141 = memref.alloc() : memref<2x128x128xf32>
    %142 = memref.alloc() : memref<128x128xf32>
    %143 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%137, %30, %arg15, %32, %33, %141, %142, %143) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %144 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%141, %144) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %145 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%144, %145) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %146 = memref.alloc() : memref<2x2x128x128xf32>
    %147 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%145, %25, %27, %146, %147) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %148 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%146, %24, %26, %148) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %149 = memref.alloc() : memref<2x2x128x64xf32>
    %150 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%148, %21, %22, %149, %150) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %151 = memref.alloc() : memref<2x128x128xf32>
    %152 = memref.alloc() : memref<128x128xf32>
    %153 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%149, %17, %arg8, %151, %152, %153) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %154 = memref.alloc() : memref<2x128x128xf32>
    %155 = memref.alloc() : memref<128x128xf32>
    %156 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%147, %17, %arg12, %154, %155, %156) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %157 = memref.alloc() : memref<2x128x128xf32>
    %158 = memref.alloc() : memref<128x128xf32>
    %159 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%150, %17, %arg10, %157, %158, %159) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %160 = call @Unknown6(%140, %151, %154, %157) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %161 = memref.alloc() : memref<2x128x128xf32>
    %162 = memref.alloc() : memref<128xf32>
    %163 = memref.alloc() : memref<128xf32>
    %164 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%160, %20, %arg6, %18, %19, %161, %162, %163, %164) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %165:2 = call @Unknown7(%7#2, %161, %11#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %166 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%84, %7#1, %165#0, %166) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %167 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%0, %11#1, %165#1, %167) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %168 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%164, %2, %168) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %169 = call @Unknown8(%14#2, %168) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    %170 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%1, %14#1, %169, %170) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    %171 = memref.alloc() : memref<30522xf32>
    "lmhlo.reduce"(%arg47, %2, %171) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %83, %166, %167, %170, %162, %163, %152, %153, %158, %159, %155, %156, %142, %143, %138, %139, %134, %135, %131, %132, %127, %128, %117, %118, %123, %124, %120, %121, %107, %108, %103, %104, %99, %100, %96, %97, %92, %93, %89, %90, %86, %87, %171 : memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

