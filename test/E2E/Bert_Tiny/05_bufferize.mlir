// RUN: byteir-opt %s -convert-hlo-to-lhlo -cse -linalg-bufferize -cse -canonicalize | FileCheck %s

// CHECK-LABEL: func @main

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0)>
module  {
  func private @MatmulOp0(%arg0: tensor<256x128xf32>, %arg1: tensor<256x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @Unknown0(%arg0: tensor<2x128xi64>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>, %arg3: tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [2, 128] : tensor<2x128xui32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%0 : tensor<2x128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %13 = trunci %arg4 : i64 to i32
      %14 = builtin.unrealized_conversion_cast %13 : i32 to ui32
      linalg.yield %14 : ui32
    } -> tensor<2x128xui32>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %3 = linalg.tensor_expand_shape %arg1 [[0, 1]] : tensor<256xi64> into tensor<2x128xi64>
    %4 = linalg.tensor_expand_shape %arg2 [[0, 1]] : tensor<256xi64> into tensor<2x128xi64>
    %5 = linalg.init_tensor [2, 128] : tensor<2x128xi64>
    %6 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3, %4 : tensor<2x128xi64>, tensor<2x128xi64>, tensor<2x128xi64>) outs(%5 : tensor<2x128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %13 = addi %arg4, %arg6 : i64
      %14 = cmpi slt, %arg4, %arg5 : i64
      %15 = select %14, %13, %arg4 : i64
      linalg.yield %15 : i64
    } -> tensor<2x128xi64>
    %7 = linalg.tensor_collapse_shape %6 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %8 = linalg.tensor_expand_shape %7 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    %9 = linalg.tensor_expand_shape %arg3 [[0, 1]] : tensor<256xf64> into tensor<2x128xf64>
    %10 = linalg.init_tensor [2, 128] : tensor<2x128xi1>
    %11 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %9 : tensor<2x128xi64>, tensor<2x128xf64>) outs(%10 : tensor<2x128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %13 = sitofp %arg4 : i64 to f64
      %14 = cmpf une, %13, %arg5 : f64
      linalg.yield %14 : i1
    } -> tensor<2x128xi1>
    %12 = linalg.tensor_collapse_shape %11 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    return %2, %8, %12 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown1(%arg0: tensor<128xi64>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>, %arg3: tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [2, 128] : tensor<2x128xui32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xi64>) outs(%0 : tensor<2x128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %13 = trunci %arg4 : i64 to i32
      %14 = builtin.unrealized_conversion_cast %13 : i32 to ui32
      linalg.yield %14 : ui32
    } -> tensor<2x128xui32>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %3 = linalg.tensor_expand_shape %arg1 [[0, 1]] : tensor<256xi64> into tensor<2x128xi64>
    %4 = linalg.tensor_expand_shape %arg2 [[0, 1]] : tensor<256xi64> into tensor<2x128xi64>
    %5 = linalg.init_tensor [2, 128] : tensor<2x128xi64>
    %6 = linalg.generic {indexing_maps = [#map1, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3, %4 : tensor<128xi64>, tensor<2x128xi64>, tensor<2x128xi64>) outs(%5 : tensor<2x128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %13 = addi %arg4, %arg6 : i64
      %14 = cmpi slt, %arg4, %arg5 : i64
      %15 = select %14, %13, %arg4 : i64
      linalg.yield %15 : i64
    } -> tensor<2x128xi64>
    %7 = linalg.tensor_collapse_shape %6 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %8 = linalg.tensor_expand_shape %7 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    %9 = linalg.tensor_expand_shape %arg3 [[0, 1]] : tensor<256xf64> into tensor<2x128xf64>
    %10 = linalg.init_tensor [2, 128] : tensor<2x128xi1>
    %11 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %9 : tensor<128xi64>, tensor<2x128xf64>) outs(%10 : tensor<2x128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %13 = sitofp %arg4 : i64 to f64
      %14 = cmpf une, %13, %arg5 : f64
      linalg.yield %14 : i1
    } -> tensor<2x128xi1>
    %12 = linalg.tensor_collapse_shape %11 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    return %2, %8, %12 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown2(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<256x128xf32> into tensor<2x128x128xf32>
    %1 = linalg.tensor_expand_shape %arg1 [[0, 1], [2]] : tensor<256x128xf32> into tensor<2x128x128xf32>
    %2 = linalg.init_tensor [2, 128, 128] : tensor<2x128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%2 : tensor<2x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %4 = addf %arg2, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func private @Unknown3(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %1 = linalg.init_tensor [128] : tensor<128xui32>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%1 : tensor<128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %8 = trunci %arg4 : i64 to i32
      %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
      linalg.yield %9 : ui32
    } -> tensor<128xui32>
    %3 = linalg.init_tensor [128] : tensor<128xi64>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3], iterator_types = ["parallel"]} ins(%0, %arg1, %arg2 : tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) outs(%3 : tensor<128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %8 = addi %arg4, %arg6 : i64
      %9 = cmpi slt, %arg4, %arg5 : i64
      %10 = select %9, %8, %arg4 : i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %5 = linalg.tensor_expand_shape %4 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %6 = linalg.init_tensor [128] : tensor<128xi1>
    %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%0, %arg3 : tensor<128xi64>, tensor<128xf64>) outs(%6 : tensor<128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %8 = sitofp %arg4 : i64 to f64
      %9 = cmpf une, %8, %arg5 : f64
      linalg.yield %9 : i1
    } -> tensor<128xi1>
    return %2, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown4(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522xf32>) -> tensor<2x128x30522xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<256x30522xf32> into tensor<2x128x30522xf32>
    %1 = linalg.init_tensor [2, 128, 30522] : tensor<2x128x30522xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : tensor<2x128x30522xf32>, tensor<30522xf32>) outs(%1 : tensor<2x128x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    } -> tensor<2x128x30522xf32>
    return %2 : tensor<2x128x30522xf32>
  }
  func private @Unknown5(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [2, 128, 128] : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %2 = addf %arg4, %arg5 : f32
      %3 = addf %2, %arg6 : f32
      %4 = addf %3, %arg7 : f32
      linalg.yield %4 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func private @Unknown6(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [2, 128, 128] : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %2 = addf %arg4, %arg5 : f32
      %3 = addf %2, %arg6 : f32
      %4 = addf %3, %arg7 : f32
      linalg.yield %4 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func private @Unknown7(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256x128xf32>, %arg3: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %1 = linalg.tensor_expand_shape %arg2 [[0, 1], [2]] : tensor<256x128xf32> into tensor<2x128x128xf32>
    %2 = linalg.init_tensor [2, 128, 128] : tensor<2x128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map5, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1, %1 : tensor<2x128xi1>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%2 : tensor<2x128x128xf32>) {
    ^bb0(%arg4: i1, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
      %8 = select %arg4, %arg5, %arg6 : f32
      linalg.yield %8 : f32
    } -> tensor<2x128x128xf32>
    %4 = linalg.tensor_collapse_shape %3 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    %5 = linalg.tensor_expand_shape %arg3 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %6 = linalg.generic {indexing_maps = [#map5, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %arg1, %1 : tensor<2x128xi1>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%2 : tensor<2x128x128xf32>) {
    ^bb0(%arg4: i1, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
      %8 = select %arg4, %arg5, %arg6 : f32
      linalg.yield %8 : f32
    } -> tensor<2x128x128xf32>
    %7 = linalg.tensor_collapse_shape %6 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    return %4, %7 : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func private @Unknown8(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [128, 128] : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map6, #map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<1x512xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<30522x128xf32>, %arg4: tensor<2x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<2x1x1x128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<2x1x1x128xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<512x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<30522xf32>, %arg47: tensor<2x128x30522xf32>) -> (tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %1 = mhlo.constant dense<30522> : tensor<256xi64>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<256xf64>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %4 = mhlo.constant dense<0> : tensor<256xi64>
    %5 = mhlo.constant dense<2> : tensor<256xi64>
    %6 = mhlo.constant dense<-1.000000e+00> : tensor<256xf64>
    %7 = mhlo.constant dense<0.000000e+00> : tensor<256x128xf32>
    %8 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %9 = mhlo.constant dense<0> : tensor<128xi64>
    %10 = mhlo.constant dense<512> : tensor<128xi64>
    %11 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %12 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %13 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %15 = "mhlo.reshape"(%14) : (tensor<1x128xi64>) -> tensor<128xi64>
    %16 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %17 = "mhlo.reshape"(%arg47) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %18:3 = call @Unknown0(%arg0, %4, %1, %2) : (tensor<2x128xi64>, tensor<256xi64>, tensor<256xi64>, tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %19 = "mhlo.gather"(%arg3, %18#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %20 = "mhlo.dot"(%17, %arg3) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %21 = "mhlo.reshape"(%20) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %22:3 = call @Unknown1(%15, %4, %5, %6) : (tensor<128xi64>, tensor<256xi64>, tensor<256xi64>, tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %23 = "mhlo.gather"(%arg4, %22#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %24 = call @Unknown2(%19, %23) : (tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %25:3 = call @Unknown3(%16, %9, %10, %11) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %26 = "mhlo.gather"(%arg5, %25#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %27 = "mhlo.reshape"(%26) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %28 = "mhlo.custom_call"(%24, %arg6, %arg7, %27) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %29 = "mhlo.get_tuple_element"(%28) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %30 = "mhlo.custom_call"(%29, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %31 = "mhlo.custom_call"(%29, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %32 = "mhlo.custom_call"(%30, %31) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %33 = "mhlo.custom_call"(%32, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %35 = "mhlo.custom_call"(%29, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %36 = "mhlo.custom_call"(%34, %35) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %37 = "mhlo.custom_call"(%36) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %38 = "mhlo.reshape"(%37) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %39 = "mhlo.custom_call"(%38, %arg15, %arg16) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %40 = "mhlo.get_tuple_element"(%39) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %41 = "mhlo.custom_call"(%40, %arg17, %arg18, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %42 = "mhlo.get_tuple_element"(%41) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %43 = "mhlo.custom_call"(%42, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %44 = "mhlo.get_tuple_element"(%43) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %45 = "mhlo.custom_call"(%44, %arg21, %arg22) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %46 = "mhlo.get_tuple_element"(%45) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %47 = "mhlo.custom_call"(%46, %arg23, %arg24, %42) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %49 = "mhlo.custom_call"(%48, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %50 = "mhlo.custom_call"(%48, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %51 = "mhlo.custom_call"(%49, %50) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %52 = "mhlo.custom_call"(%51, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %53 = "mhlo.get_tuple_element"(%52) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %54 = "mhlo.custom_call"(%48, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %55 = "mhlo.custom_call"(%53, %54) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %56 = "mhlo.custom_call"(%55) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %57 = "mhlo.reshape"(%56) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %58 = "mhlo.custom_call"(%57, %arg32, %arg33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %59 = "mhlo.get_tuple_element"(%58) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %60 = "mhlo.custom_call"(%59, %arg34, %arg35, %48) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %61 = "mhlo.get_tuple_element"(%60) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %62 = "mhlo.custom_call"(%61, %arg36, %arg37) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %63 = "mhlo.get_tuple_element"(%62) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %64 = "mhlo.custom_call"(%63, %arg38, %arg39) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %65 = "mhlo.get_tuple_element"(%64) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %66 = "mhlo.custom_call"(%65, %arg40, %arg41, %61) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %67 = "mhlo.get_tuple_element"(%66) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %68 = "mhlo.custom_call"(%67, %arg42, %arg43) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %69 = "mhlo.get_tuple_element"(%68) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %70 = "mhlo.custom_call"(%69, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %71 = "mhlo.get_tuple_element"(%70) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %72 = "mhlo.reshape"(%71) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %73 = "mhlo.dot_general"(%72, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<30522x128xf32>) -> tensor<256x30522xf32>
    %74 = call @Unknown4(%73, %arg46) : (tensor<256x30522xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %75 = call @MatmulOp0(%72, %17) : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<30522x128xf32>
    %76 = "mhlo.get_tuple_element"(%70) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %77 = "mhlo.get_tuple_element"(%70) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %78 = "mhlo.custom_call"(%21, %69, %arg44, %76, %77) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %79 = "mhlo.get_tuple_element"(%78) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %80 = "mhlo.get_tuple_element"(%68) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %81 = "mhlo.get_tuple_element"(%68) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %82 = "mhlo.custom_call"(%79, %67, %arg42, %80, %81) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %83 = "mhlo.get_tuple_element"(%82) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %84 = "mhlo.get_tuple_element"(%66) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %85 = "mhlo.get_tuple_element"(%66) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %86 = "mhlo.get_tuple_element"(%66) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %87 = "mhlo.custom_call"(%83, %84, %arg40, %85, %86) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %88 = "mhlo.get_tuple_element"(%87) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %89 = "mhlo.get_tuple_element"(%87) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %90 = "mhlo.get_tuple_element"(%64) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %91 = "mhlo.get_tuple_element"(%64) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %92 = "mhlo.custom_call"(%89, %63, %arg38, %90, %91) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %93 = "mhlo.get_tuple_element"(%92) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %94 = "mhlo.get_tuple_element"(%62) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %95 = "mhlo.get_tuple_element"(%62) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %96 = "mhlo.custom_call"(%93, %61, %arg36, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %97 = "mhlo.get_tuple_element"(%96) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %98 = mhlo.add %88, %97 : tensor<2x128x128xf32>
    %99 = "mhlo.get_tuple_element"(%60) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %100 = "mhlo.get_tuple_element"(%60) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %101 = "mhlo.get_tuple_element"(%60) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %102 = "mhlo.custom_call"(%98, %99, %arg34, %100, %101) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %103 = "mhlo.get_tuple_element"(%102) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %104 = "mhlo.get_tuple_element"(%102) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %105 = "mhlo.get_tuple_element"(%58) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %106 = "mhlo.get_tuple_element"(%58) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %107 = "mhlo.custom_call"(%104, %57, %arg32, %105, %106) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %108 = "mhlo.get_tuple_element"(%107) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %109 = "mhlo.reshape"(%108) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %110 = "mhlo.custom_call"(%109) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %111 = "mhlo.custom_call"(%110, %53, %54) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %112 = "mhlo.get_tuple_element"(%111) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %113 = "mhlo.get_tuple_element"(%52) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %114 = "mhlo.get_tuple_element"(%52) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %115 = "mhlo.custom_call"(%112, %113, %114) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %116 = "mhlo.custom_call"(%115, %49, %50) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %117 = "mhlo.get_tuple_element"(%116) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %118 = "mhlo.custom_call"(%117, %48, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %120 = "mhlo.get_tuple_element"(%111) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %121 = "mhlo.custom_call"(%120, %48, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %122 = "mhlo.get_tuple_element"(%121) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %123 = "mhlo.get_tuple_element"(%116) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %124 = "mhlo.custom_call"(%123, %48, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %125 = "mhlo.get_tuple_element"(%124) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %126 = call @Unknown5(%103, %119, %122, %125) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %127 = "mhlo.get_tuple_element"(%47) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %128 = "mhlo.get_tuple_element"(%47) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %129 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %130 = "mhlo.custom_call"(%126, %127, %arg23, %128, %129) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %131 = "mhlo.get_tuple_element"(%130) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %132 = "mhlo.get_tuple_element"(%130) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %133 = "mhlo.get_tuple_element"(%45) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %134 = "mhlo.get_tuple_element"(%45) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %135 = "mhlo.custom_call"(%132, %44, %arg21, %133, %134) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %136 = "mhlo.get_tuple_element"(%135) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %137 = "mhlo.get_tuple_element"(%43) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %138 = "mhlo.get_tuple_element"(%43) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %139 = "mhlo.custom_call"(%136, %42, %arg19, %137, %138) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %140 = "mhlo.get_tuple_element"(%139) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %141 = mhlo.add %131, %140 : tensor<2x128x128xf32>
    %142 = "mhlo.get_tuple_element"(%41) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %143 = "mhlo.get_tuple_element"(%41) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %144 = "mhlo.get_tuple_element"(%41) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %145 = "mhlo.custom_call"(%141, %142, %arg17, %143, %144) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %146 = "mhlo.get_tuple_element"(%145) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %147 = "mhlo.get_tuple_element"(%145) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %148 = "mhlo.get_tuple_element"(%39) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %149 = "mhlo.get_tuple_element"(%39) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %150 = "mhlo.custom_call"(%147, %38, %arg15, %148, %149) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %151 = "mhlo.get_tuple_element"(%150) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %152 = "mhlo.reshape"(%151) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %153 = "mhlo.custom_call"(%152) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %154 = "mhlo.custom_call"(%153, %34, %35) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %155 = "mhlo.get_tuple_element"(%154) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %156 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %157 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %158 = "mhlo.custom_call"(%155, %156, %157) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %159 = "mhlo.custom_call"(%158, %30, %31) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %160 = "mhlo.get_tuple_element"(%159) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %161 = "mhlo.custom_call"(%160, %29, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %162 = "mhlo.get_tuple_element"(%161) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %163 = "mhlo.get_tuple_element"(%154) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %164 = "mhlo.custom_call"(%163, %29, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %165 = "mhlo.get_tuple_element"(%164) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %166 = "mhlo.get_tuple_element"(%159) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %167 = "mhlo.custom_call"(%166, %29, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %168 = "mhlo.get_tuple_element"(%167) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %169 = call @Unknown6(%146, %162, %165, %168) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %170 = "mhlo.get_tuple_element"(%28) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %171 = "mhlo.get_tuple_element"(%28) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %172 = "mhlo.get_tuple_element"(%28) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %173 = "mhlo.custom_call"(%169, %170, %arg6, %171, %172) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %174 = "mhlo.get_tuple_element"(%173) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %175:2 = call @Unknown7(%18#2, %174, %7, %22#2) : (tensor<256xi1>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>)
    %176 = "mhlo.scatter"(%0, %18#1, %175#0) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %222 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%222) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %177 = mhlo.add %75, %176 {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : tensor<30522x128xf32>
    %178 = "mhlo.scatter"(%3, %22#1, %175#1) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %222 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%222) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %179 = "mhlo.get_tuple_element"(%173) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %180 = "mhlo.reduce"(%179, %13) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %222 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%222) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %181 = call @Unknown8(%25#2, %180, %12) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %182 = "mhlo.scatter"(%8, %25#1, %181) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %222 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%222) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %183 = "mhlo.get_tuple_element"(%173) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %184 = "mhlo.get_tuple_element"(%173) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %185 = "mhlo.get_tuple_element"(%161) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %186 = "mhlo.get_tuple_element"(%161) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %187 = "mhlo.get_tuple_element"(%167) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %188 = "mhlo.get_tuple_element"(%167) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %189 = "mhlo.get_tuple_element"(%164) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %190 = "mhlo.get_tuple_element"(%164) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %191 = "mhlo.get_tuple_element"(%150) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %192 = "mhlo.get_tuple_element"(%150) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %193 = "mhlo.get_tuple_element"(%145) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %194 = "mhlo.get_tuple_element"(%145) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %195 = "mhlo.get_tuple_element"(%139) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %196 = "mhlo.get_tuple_element"(%139) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %197 = "mhlo.get_tuple_element"(%135) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %198 = "mhlo.get_tuple_element"(%135) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %199 = "mhlo.get_tuple_element"(%130) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %200 = "mhlo.get_tuple_element"(%130) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %201 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %202 = "mhlo.get_tuple_element"(%118) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %203 = "mhlo.get_tuple_element"(%124) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %204 = "mhlo.get_tuple_element"(%124) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %205 = "mhlo.get_tuple_element"(%121) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %206 = "mhlo.get_tuple_element"(%121) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %207 = "mhlo.get_tuple_element"(%107) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %208 = "mhlo.get_tuple_element"(%107) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %209 = "mhlo.get_tuple_element"(%102) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %210 = "mhlo.get_tuple_element"(%102) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %211 = "mhlo.get_tuple_element"(%96) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %212 = "mhlo.get_tuple_element"(%96) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %213 = "mhlo.get_tuple_element"(%92) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %214 = "mhlo.get_tuple_element"(%92) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %215 = "mhlo.get_tuple_element"(%87) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %216 = "mhlo.get_tuple_element"(%87) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %217 = "mhlo.get_tuple_element"(%82) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %218 = "mhlo.get_tuple_element"(%82) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %219 = "mhlo.get_tuple_element"(%78) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %220 = "mhlo.get_tuple_element"(%78) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %221 = "mhlo.reduce"(%arg47, %13) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %222 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%222) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    return %74, %177, %178, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %205, %206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %216, %217, %218, %219, %220, %221 : tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>
  }
}
