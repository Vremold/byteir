// RUN: byteir-opt %s -convert-linalg-to-affine-loops -loop-coalescing -simplify-affine-structures -affine-loop-fusion -affine-loop-fusion-ex -cse -cmae | FileCheck %s

// CHECK-LABEL: func @main

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module  {
  func private @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128xui32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%0 : memref<2x128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %10 = trunci %arg4 : i64 to i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      linalg.yield %11 : ui32
    }
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %2 = memref.expand_shape %arg1 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %3 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %4 = memref.alloc() : memref<2x128xi64>
    linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2, %3 : memref<2x128xi64>, memref<2x128xi64>, memref<2x128xi64>) outs(%4 : memref<2x128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %10 = addi %arg4, %arg6 : i64
      %11 = cmpi slt, %arg4, %arg5 : i64
      %12 = select %11, %10, %arg4 : i64
      linalg.yield %12 : i64
    }
    %5 = memref.collapse_shape %4 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %6 = memref.expand_shape %5 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %7 = memref.expand_shape %arg3 [[0, 1]] : memref<256xf64> into memref<2x128xf64>
    %8 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %7 : memref<2x128xi64>, memref<2x128xf64>) outs(%8 : memref<2x128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %10 = sitofp %arg4 : i64 to f64
      %11 = cmpf une, %10, %arg5 : f64
      linalg.yield %11 : i1
    }
    %9 = memref.collapse_shape %8 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %1, %6, %9 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown1(%arg0: memref<128xi64>, %arg1: memref<256xi64>, %arg2: memref<256xi64>, %arg3: memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128xui32>
    linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%0 : memref<2x128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %10 = trunci %arg4 : i64 to i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      linalg.yield %11 : ui32
    }
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %2 = memref.expand_shape %arg1 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %3 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi64> into memref<2x128xi64>
    %4 = memref.alloc() : memref<2x128xi64>
    linalg.generic {indexing_maps = [#map1, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2, %3 : memref<128xi64>, memref<2x128xi64>, memref<2x128xi64>) outs(%4 : memref<2x128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %10 = addi %arg4, %arg6 : i64
      %11 = cmpi slt, %arg4, %arg5 : i64
      %12 = select %11, %10, %arg4 : i64
      linalg.yield %12 : i64
    }
    %5 = memref.collapse_shape %4 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %6 = memref.expand_shape %5 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %7 = memref.expand_shape %arg3 [[0, 1]] : memref<256xf64> into memref<2x128xf64>
    %8 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %7 : memref<128xi64>, memref<2x128xf64>) outs(%8 : memref<2x128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %10 = sitofp %arg4 : i64 to f64
      %11 = cmpf une, %10, %arg5 : f64
      linalg.yield %11 : i1
    }
    %9 = memref.collapse_shape %8 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %1, %6, %9 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func private @Unknown2(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %1 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%2 : memref<2x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    }
    return %2 : memref<2x128x128xf32>
  }
  func private @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<128xi64>, %arg2: memref<128xi64>, %arg3: memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %1 = memref.alloc() : memref<128xui32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : memref<128xi64>) outs(%1 : memref<128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %5 = trunci %arg4 : i64 to i32
      %6 = builtin.unrealized_conversion_cast %5 : i32 to ui32
      linalg.yield %6 : ui32
    }
    %2 = memref.alloc() : memref<128xi64>
    linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel"]} ins(%0, %arg1, %arg2 : memref<128xi64>, memref<128xi64>, memref<128xi64>) outs(%2 : memref<128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %5 = addi %arg4, %arg6 : i64
      %6 = cmpi slt, %arg4, %arg5 : i64
      %7 = select %6, %5, %arg4 : i64
      linalg.yield %7 : i64
    }
    %3 = memref.expand_shape %2 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %4 = memref.alloc() : memref<128xi1>
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%0, %arg3 : memref<128xi64>, memref<128xf64>) outs(%4 : memref<128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %5 = sitofp %arg4 : i64 to f64
      %6 = cmpf une, %5, %arg5 : f64
      linalg.yield %6 : i1
    }
    return %1, %3, %4 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func private @Unknown4(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %1 = addf %arg4, %arg5 : f32
      %2 = addf %1, %arg6 : f32
      %3 = addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown5(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%0 : memref<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %1 = addf %arg4, %arg5 : f32
      %2 = addf %1, %arg6 : f32
      %3 = addf %2, %arg7 : f32
      linalg.yield %3 : f32
    }
    return %0 : memref<2x128x128xf32>
  }
  func private @Unknown6(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256x128xf32>, %arg3: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %1 = memref.expand_shape %arg2 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map4, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1, %1 : memref<2x128xi1>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%2 : memref<2x128x128xf32>) {
    ^bb0(%arg4: i1, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
      %7 = select %arg4, %arg5, %arg6 : f32
      linalg.yield %7 : f32
    }
    %3 = memref.collapse_shape %2 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %4 = memref.expand_shape %arg3 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %5 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map4, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %arg1, %1 : memref<2x128xi1>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%5 : memref<2x128x128xf32>) {
    ^bb0(%arg4: i1, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
      %7 = select %arg4, %arg5, %arg6 : f32
      linalg.yield %7 : f32
    }
    %6 = memref.collapse_shape %5 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    return %3, %6 : memref<256x128xf32>, memref<256x128xf32>
  }
  func private @Unknown7(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) -> memref<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = memref.alloc() : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map5, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>) outs(%0 : memref<128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %1 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %1 : f32
    }
    return %0 : memref<128x128xf32>
  }
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<1x512xi64>, %arg2: memref<1x512xi64>, %arg3: memref<30522x128xf32>, %arg4: memref<2x128xf32>, %arg5: memref<512x128xf32>, %arg6: memref<128xf32>, %arg7: memref<128xf32>, %arg8: memref<128x128xf32>, %arg9: memref<128xf32>, %arg10: memref<128x128xf32>, %arg11: memref<128xf32>, %arg12: memref<128x128xf32>, %arg13: memref<128xf32>, %arg14: memref<2x1x1x128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<2x1x1x128xf32>, %arg32: memref<128x128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<512x128xf32>, %arg37: memref<512xf32>, %arg38: memref<128x512xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<30522xf32>, %arg47: memref<2x128x30522xf32>) -> (memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<128x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<128x128xf32>} : (memref<128x128xf32>) -> ()
    %1 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %2 = memref.alloc() : memref<128xf64>
    "lmhlo.constant"(%2) {value = dense<-1.000000e+00> : tensor<128xf64>} : (memref<128xf64>) -> ()
    %3 = memref.alloc() : memref<128xi64>
    "lmhlo.constant"(%3) {value = dense<512> : tensor<128xi64>} : (memref<128xi64>) -> ()
    %4 = memref.alloc() : memref<128xi64>
    "lmhlo.constant"(%4) {value = dense<0> : tensor<128xi64>} : (memref<128xi64>) -> ()
    %5 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%5) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %6 = memref.alloc() : memref<256x128xf32>
    "lmhlo.constant"(%6) {value = dense<0.000000e+00> : tensor<256x128xf32>} : (memref<256x128xf32>) -> ()
    %7 = memref.alloc() : memref<256xf64>
    "lmhlo.constant"(%7) {value = dense<-1.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    %8 = memref.alloc() : memref<256xi64>
    "lmhlo.constant"(%8) {value = dense<2> : tensor<256xi64>} : (memref<256xi64>) -> ()
    %9 = memref.alloc() : memref<256xi64>
    "lmhlo.constant"(%9) {value = dense<0> : tensor<256xi64>} : (memref<256xi64>) -> ()
    %10 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%10) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %11 = memref.alloc() : memref<256xf64>
    "lmhlo.constant"(%11) {value = dense<0.000000e+00> : tensor<256xf64>} : (memref<256xf64>) -> ()
    %12 = memref.alloc() : memref<256xi64>
    "lmhlo.constant"(%12) {value = dense<30522> : tensor<256xi64>} : (memref<256xi64>) -> ()
    %13 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.constant"(%13) {value = dense<0.000000e+00> : tensor<30522x128xf32>} : (memref<30522x128xf32>) -> ()
    %14 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg1, %14) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %15 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%14, %15) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %16 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %16) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %17:3 = call @Unknown0(%arg0, %9, %12, %11) : (memref<2x128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %18 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg3, %17#0, %18) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %19:3 = call @Unknown1(%15, %9, %8, %7) : (memref<128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %20 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %19#0, %20) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %21 = call @Unknown2(%18, %20) : (memref<256x128xf32>, memref<256x128xf32>) -> memref<2x128x128xf32>
    %22:3 = call @Unknown3(%16, %4, %3, %2) : (memref<1x128xi64>, memref<128xi64>, memref<128xi64>, memref<128xf64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %23 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg5, %22#0, %23) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %24 = memref.alloc() : memref<1x128x128xf32>
    "lmhlo.reshape"(%23, %24) : (memref<128x128xf32>, memref<1x128x128xf32>) -> ()
    %25 = memref.alloc() : memref<2x128x128xf32>
    %26 = memref.alloc() : memref<256xf32>
    %27 = memref.alloc() : memref<256xf32>
    %28 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%21, %arg6, %arg7, %24, %25, %26, %27, %28) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %29 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%25, %arg8, %arg9, %29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %30 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%25, %arg10, %arg11, %30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %31 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%29, %30, %31) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %32 = memref.alloc() : memref<2x2x128x128xf32>
    %33 = memref.alloc() : memref<2x2x128x128xf32>
    %34 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%31, %arg14, %32, %33, %34) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %35 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%25, %arg12, %arg13, %35) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %36 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%33, %35, %36) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %37 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%36, %37) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %38 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%37, %38) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %39 = memref.alloc() : memref<2x128x128xf32>
    %40 = memref.alloc() : memref<2x128x128xf32>
    %41 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%38, %arg15, %arg16, %39, %40, %41) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %42 = memref.alloc() : memref<2x128x128xf32>
    %43 = memref.alloc() : memref<256xf32>
    %44 = memref.alloc() : memref<256xf32>
    %45 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%39, %arg17, %arg18, %25, %42, %43, %44, %45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %46 = memref.alloc() : memref<2x128x512xf32>
    %47 = memref.alloc() : memref<2x128x512xf32>
    %48 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%42, %arg19, %arg20, %46, %47, %48) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %49 = memref.alloc() : memref<2x128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%46, %arg21, %arg22, %49, %50, %51) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %52 = memref.alloc() : memref<2x128x128xf32>
    %53 = memref.alloc() : memref<256xf32>
    %54 = memref.alloc() : memref<256xf32>
    %55 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%49, %arg23, %arg24, %42, %52, %53, %54, %55) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %56 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%52, %arg25, %arg26, %56) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %57 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%52, %arg27, %arg28, %57) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %58 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%56, %57, %58) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %59 = memref.alloc() : memref<2x2x128x128xf32>
    %60 = memref.alloc() : memref<2x2x128x128xf32>
    %61 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%58, %arg31, %59, %60, %61) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %62 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%52, %arg29, %arg30, %62) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %63 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%60, %62, %63) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %64 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%63, %64) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %65 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%64, %65) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%65, %arg32, %arg33, %66, %67, %68) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %69 = memref.alloc() : memref<2x128x128xf32>
    %70 = memref.alloc() : memref<256xf32>
    %71 = memref.alloc() : memref<256xf32>
    %72 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%66, %arg34, %arg35, %52, %69, %70, %71, %72) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %73 = memref.alloc() : memref<2x128x512xf32>
    %74 = memref.alloc() : memref<2x128x512xf32>
    %75 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%69, %arg36, %arg37, %73, %74, %75) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %76 = memref.alloc() : memref<2x128x128xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    %78 = memref.alloc() : memref<2x128x128xui8>
    "lmhlo.custom_call"(%73, %arg38, %arg39, %76, %77, %78) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    %79 = memref.alloc() : memref<2x128x128xf32>
    %80 = memref.alloc() : memref<256xf32>
    %81 = memref.alloc() : memref<256xf32>
    %82 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%76, %arg40, %arg41, %69, %79, %80, %81, %82) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %83 = memref.alloc() : memref<2x128x128xf32>
    %84 = memref.alloc() : memref<2x128x128xf32>
    %85 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%79, %arg42, %arg43, %83, %84, %85) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %86 = memref.alloc() : memref<2x128x128xf32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%83, %arg44, %arg45, %86, %87, %88) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %89 = memref.alloc() : memref<2x128x30522xf32>
    "lmhlo.custom_call"(%86, %arg3, %arg46, %89) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>, memref<2x128x30522xf32>) -> ()
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<30522x128xf32>
    %92 = memref.alloc() : memref<30522xf32>
    "lmhlo.custom_call"(%arg47, %86, %arg3, %90, %91, %92) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x30522xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>) -> ()
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<128xf32>
    %95 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%90, %83, %arg44, %87, %88, %93, %94, %95) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<128x128xf32>
    %98 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%93, %79, %arg42, %84, %85, %96, %97, %98) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %99 = memref.alloc() : memref<2x128x128xf32>
    %100 = memref.alloc() : memref<128xf32>
    %101 = memref.alloc() : memref<128xf32>
    %102 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%96, %82, %arg40, %80, %81, %99, %100, %101, %102) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %103 = memref.alloc() : memref<2x128x512xf32>
    %104 = memref.alloc() : memref<128x512xf32>
    %105 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%99, %73, %arg38, %77, %78, %103, %104, %105) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %106 = memref.alloc() : memref<2x128x128xf32>
    %107 = memref.alloc() : memref<512x128xf32>
    %108 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%103, %69, %arg36, %74, %75, %106, %107, %108) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %109 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.add"(%102, %106, %109) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    %110 = memref.alloc() : memref<2x128x128xf32>
    %111 = memref.alloc() : memref<128xf32>
    %112 = memref.alloc() : memref<128xf32>
    %113 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%109, %72, %arg34, %70, %71, %110, %111, %112, %113) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %114 = memref.alloc() : memref<2x128x128xf32>
    %115 = memref.alloc() : memref<128x128xf32>
    %116 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%110, %65, %arg32, %67, %68, %114, %115, %116) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %117 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%114, %117) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %118 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%117, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %119 = memref.alloc() : memref<2x2x128x128xf32>
    %120 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%118, %60, %62, %119, %120) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %121 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%119, %59, %61, %121) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %122 = memref.alloc() : memref<2x2x128x64xf32>
    %123 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%121, %56, %57, %122, %123) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<128x128xf32>
    %126 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%122, %52, %arg25, %124, %125, %126) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %127 = memref.alloc() : memref<2x128x128xf32>
    %128 = memref.alloc() : memref<128x128xf32>
    %129 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%120, %52, %arg29, %127, %128, %129) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %130 = memref.alloc() : memref<2x128x128xf32>
    %131 = memref.alloc() : memref<128x128xf32>
    %132 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%123, %52, %arg27, %130, %131, %132) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %133 = call @Unknown4(%113, %124, %127, %130) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %134 = memref.alloc() : memref<2x128x128xf32>
    %135 = memref.alloc() : memref<128xf32>
    %136 = memref.alloc() : memref<128xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%133, %55, %arg23, %53, %54, %134, %135, %136, %137) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %138 = memref.alloc() : memref<2x128x512xf32>
    %139 = memref.alloc() : memref<128x512xf32>
    %140 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%134, %46, %arg21, %50, %51, %138, %139, %140) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %141 = memref.alloc() : memref<2x128x128xf32>
    %142 = memref.alloc() : memref<512x128xf32>
    %143 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%138, %42, %arg19, %47, %48, %141, %142, %143) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %144 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.add"(%137, %141, %144) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    %145 = memref.alloc() : memref<2x128x128xf32>
    %146 = memref.alloc() : memref<128xf32>
    %147 = memref.alloc() : memref<128xf32>
    %148 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%144, %45, %arg17, %43, %44, %145, %146, %147, %148) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %149 = memref.alloc() : memref<2x128x128xf32>
    %150 = memref.alloc() : memref<128x128xf32>
    %151 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%145, %38, %arg15, %40, %41, %149, %150, %151) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %152 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%149, %152) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %153 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%152, %153) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %154 = memref.alloc() : memref<2x2x128x128xf32>
    %155 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%153, %33, %35, %154, %155) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %156 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%154, %32, %34, %156) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %157 = memref.alloc() : memref<2x2x128x64xf32>
    %158 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%156, %29, %30, %157, %158) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %159 = memref.alloc() : memref<2x128x128xf32>
    %160 = memref.alloc() : memref<128x128xf32>
    %161 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%157, %25, %arg8, %159, %160, %161) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %162 = memref.alloc() : memref<2x128x128xf32>
    %163 = memref.alloc() : memref<128x128xf32>
    %164 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%155, %25, %arg12, %162, %163, %164) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %165 = memref.alloc() : memref<2x128x128xf32>
    %166 = memref.alloc() : memref<128x128xf32>
    %167 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%158, %25, %arg10, %165, %166, %167) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %168 = call @Unknown5(%148, %159, %162, %165) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %169 = memref.alloc() : memref<2x128x128xf32>
    %170 = memref.alloc() : memref<128xf32>
    %171 = memref.alloc() : memref<128xf32>
    %172 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%168, %28, %arg6, %26, %27, %169, %170, %171, %172) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %173:2 = call @Unknown6(%17#2, %169, %6, %19#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %174 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%13, %17#1, %173#0, %174) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %175 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.add"(%91, %174, %175) : (memref<30522x128xf32>, memref<30522x128xf32>, memref<30522x128xf32>) -> ()
    %176 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%10, %19#1, %173#1, %176) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %177 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%172, %1, %177) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %178 = call @Unknown7(%22#2, %177, %0) : (memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>) -> memref<128x128xf32>
    %179 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%5, %22#1, %178, %179) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %180 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%180) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    return %89, %175, %176, %179, %170, %171, %160, %161, %166, %167, %163, %164, %150, %151, %146, %147, %142, %143, %139, %140, %135, %136, %125, %126, %131, %132, %128, %129, %115, %116, %111, %112, %107, %108, %104, %105, %100, %101, %97, %98, %94, %95, %92 : memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

