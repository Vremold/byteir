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
  func private @Unknown0(%arg0: tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %cst = constant 0.000000e+00 : f64
    %c30522_i64 = constant 30522 : i64
    %c0_i64 = constant 0 : i64
    %0 = linalg.init_tensor [2, 128] : tensor<2x128xui32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%0 : tensor<2x128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %10 = trunci %arg1 : i64 to i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      linalg.yield %11 : ui32
    } -> tensor<2x128xui32>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %3 = linalg.init_tensor [2, 128] : tensor<2x128xi64>
    %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%3 : tensor<2x128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %10 = addi %arg1, %c30522_i64 : i64
      %11 = cmpi slt, %arg1, %c0_i64 : i64
      %12 = select %11, %10, %arg1 : i64
      linalg.yield %12 : i64
    } -> tensor<2x128xi64>
    %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %6 = linalg.tensor_expand_shape %5 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    %7 = linalg.init_tensor [2, 128] : tensor<2x128xi1>
    %8 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%7 : tensor<2x128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %10 = sitofp %arg1 : i64 to f64
      %11 = cmpf une, %10, %cst : f64
      linalg.yield %11 : i1
    } -> tensor<2x128xi1>
    %9 = linalg.tensor_collapse_shape %8 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    return %2, %6, %9 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown1(%arg0: tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %cst = constant -1.000000e+00 : f64
    %c2_i64 = constant 2 : i64
    %c0_i64 = constant 0 : i64
    %0 = linalg.init_tensor [2, 128] : tensor<2x128xui32>
    %1 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xi64>) outs(%0 : tensor<2x128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %10 = trunci %arg1 : i64 to i32
      %11 = builtin.unrealized_conversion_cast %10 : i32 to ui32
      linalg.yield %11 : ui32
    } -> tensor<2x128xui32>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %3 = linalg.init_tensor [2, 128] : tensor<2x128xi64>
    %4 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xi64>) outs(%3 : tensor<2x128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %10 = addi %arg1, %c2_i64 : i64
      %11 = cmpi slt, %arg1, %c0_i64 : i64
      %12 = select %11, %10, %arg1 : i64
      linalg.yield %12 : i64
    } -> tensor<2x128xi64>
    %5 = linalg.tensor_collapse_shape %4 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %6 = linalg.tensor_expand_shape %5 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    %7 = linalg.init_tensor [2, 128] : tensor<2x128xi1>
    %8 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xi64>) outs(%7 : tensor<2x128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %10 = sitofp %arg1 : i64 to f64
      %11 = cmpf une, %10, %cst : f64
      linalg.yield %11 : i1
    } -> tensor<2x128xi1>
    %9 = linalg.tensor_collapse_shape %8 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    return %2, %6, %9 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
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
  func private @Unknown3(%arg0: tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %cst = constant -1.000000e+00 : f64
    %c512_i64 = constant 512 : i64
    %c0_i64 = constant 0 : i64
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %1 = linalg.init_tensor [128] : tensor<128xui32>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%1 : tensor<128xui32>) {
    ^bb0(%arg1: i64, %arg2: ui32):  // no predecessors
      %8 = trunci %arg1 : i64 to i32
      %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
      linalg.yield %9 : ui32
    } -> tensor<128xui32>
    %3 = linalg.init_tensor [128] : tensor<128xi64>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%3 : tensor<128xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):  // no predecessors
      %8 = addi %arg1, %c512_i64 : i64
      %9 = cmpi slt, %arg1, %c0_i64 : i64
      %10 = select %9, %8, %arg1 : i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %5 = linalg.tensor_expand_shape %4 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %6 = linalg.init_tensor [128] : tensor<128xi1>
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%6 : tensor<128xi1>) {
    ^bb0(%arg1: i64, %arg2: i1):  // no predecessors
      %8 = sitofp %arg1 : i64 to f64
      %9 = cmpf une, %8, %cst : f64
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
  func private @Unknown7(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {byre_elementwise_fusion} {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %1 = linalg.init_tensor [2, 128, 128] : tensor<2x128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map5, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : tensor<2x128xi1>, tensor<2x128x128xf32>) outs(%1 : tensor<2x128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):  // no predecessors
      %7 = select %arg3, %arg4, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<2x128x128xf32>
    %3 = linalg.tensor_collapse_shape %2 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    %4 = linalg.tensor_expand_shape %arg2 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %5 = linalg.generic {indexing_maps = [#map5, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %arg1 : tensor<2x128xi1>, tensor<2x128x128xf32>) outs(%1 : tensor<2x128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32):  // no predecessors
      %7 = select %arg3, %arg4, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<2x128x128xf32>
    %6 = linalg.tensor_collapse_shape %5 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    return %3, %6 : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func private @Unknown8(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [128, 128] : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map6, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<128xi1>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%arg2: i1, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = select %arg2, %arg3, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<1x512xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<30522x128xf32>, %arg4: tensor<2x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<2x1x1x128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<2x1x1x128xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<512x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<30522xf32>, %arg47: tensor<2x128x30522xf32>) -> (tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %4 = "mhlo.reshape"(%3) : (tensor<1x128xi64>) -> tensor<128xi64>
    %5 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %6 = "mhlo.reshape"(%arg47) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %7:3 = call @Unknown0(%arg0) : (tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %8 = "mhlo.gather"(%arg3, %7#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %9 = "mhlo.dot"(%6, %arg3) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %10 = "mhlo.reshape"(%9) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %11:3 = call @Unknown1(%4) : (tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %12 = "mhlo.gather"(%arg4, %11#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %13 = call @Unknown2(%8, %12) : (tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %14:3 = call @Unknown3(%5) : (tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %15 = "mhlo.gather"(%arg5, %14#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %16 = "mhlo.reshape"(%15) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %17 = "mhlo.custom_call"(%13, %arg6, %arg7, %16) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %18 = "mhlo.get_tuple_element"(%17) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %19 = "mhlo.custom_call"(%18, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %20 = "mhlo.custom_call"(%18, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %21 = "mhlo.custom_call"(%19, %20) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %22 = "mhlo.custom_call"(%21, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %23 = "mhlo.get_tuple_element"(%22) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %24 = "mhlo.custom_call"(%18, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %25 = "mhlo.custom_call"(%23, %24) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %26 = "mhlo.custom_call"(%25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %27 = "mhlo.reshape"(%26) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %28 = "mhlo.custom_call"(%27, %arg15, %arg16) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %29 = "mhlo.get_tuple_element"(%28) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %30 = "mhlo.custom_call"(%29, %arg17, %arg18, %18) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %31 = "mhlo.get_tuple_element"(%30) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %32 = "mhlo.custom_call"(%31, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %33 = "mhlo.get_tuple_element"(%32) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %34 = "mhlo.custom_call"(%33, %arg21, %arg22) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %35 = "mhlo.get_tuple_element"(%34) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %36 = "mhlo.custom_call"(%35, %arg23, %arg24, %31) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %37 = "mhlo.get_tuple_element"(%36) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %38 = "mhlo.custom_call"(%37, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %39 = "mhlo.custom_call"(%37, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %40 = "mhlo.custom_call"(%38, %39) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %41 = "mhlo.custom_call"(%40, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %42 = "mhlo.get_tuple_element"(%41) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %43 = "mhlo.custom_call"(%37, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %44 = "mhlo.custom_call"(%42, %43) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %45 = "mhlo.custom_call"(%44) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %46 = "mhlo.reshape"(%45) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %47 = "mhlo.custom_call"(%46, %arg32, %arg33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %49 = "mhlo.custom_call"(%48, %arg34, %arg35, %37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %50 = "mhlo.get_tuple_element"(%49) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %51 = "mhlo.custom_call"(%50, %arg36, %arg37) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %52 = "mhlo.get_tuple_element"(%51) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %53 = "mhlo.custom_call"(%52, %arg38, %arg39) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %54 = "mhlo.get_tuple_element"(%53) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %55 = "mhlo.custom_call"(%54, %arg40, %arg41, %50) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %56 = "mhlo.get_tuple_element"(%55) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %57 = "mhlo.custom_call"(%56, %arg42, %arg43) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %58 = "mhlo.get_tuple_element"(%57) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %59 = "mhlo.custom_call"(%58, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %60 = "mhlo.get_tuple_element"(%59) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %61 = "mhlo.reshape"(%60) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %62 = "mhlo.dot_general"(%61, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<30522x128xf32>) -> tensor<256x30522xf32>
    %63 = call @Unknown4(%62, %arg46) : (tensor<256x30522xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %64 = call @MatmulOp0(%61, %6) : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<30522x128xf32>
    %65 = "mhlo.get_tuple_element"(%59) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %66 = "mhlo.get_tuple_element"(%59) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %67 = "mhlo.custom_call"(%10, %58, %arg44, %65, %66) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %68 = "mhlo.get_tuple_element"(%67) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %69 = "mhlo.get_tuple_element"(%57) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %70 = "mhlo.get_tuple_element"(%57) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %71 = "mhlo.custom_call"(%68, %56, %arg42, %69, %70) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %72 = "mhlo.get_tuple_element"(%71) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %73 = "mhlo.get_tuple_element"(%55) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %74 = "mhlo.get_tuple_element"(%55) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %75 = "mhlo.get_tuple_element"(%55) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %76 = "mhlo.custom_call"(%72, %73, %arg40, %74, %75) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %77 = "mhlo.get_tuple_element"(%76) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %78 = "mhlo.get_tuple_element"(%76) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %79 = "mhlo.get_tuple_element"(%53) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %80 = "mhlo.get_tuple_element"(%53) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %81 = "mhlo.custom_call"(%78, %52, %arg38, %79, %80) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %82 = "mhlo.get_tuple_element"(%81) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %83 = "mhlo.get_tuple_element"(%51) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %84 = "mhlo.get_tuple_element"(%51) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %85 = "mhlo.custom_call"(%82, %50, %arg36, %83, %84) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %86 = "mhlo.get_tuple_element"(%85) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %87 = mhlo.add %77, %86 : tensor<2x128x128xf32>
    %88 = "mhlo.get_tuple_element"(%49) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %89 = "mhlo.get_tuple_element"(%49) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %90 = "mhlo.get_tuple_element"(%49) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %91 = "mhlo.custom_call"(%87, %88, %arg34, %89, %90) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %92 = "mhlo.get_tuple_element"(%91) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %93 = "mhlo.get_tuple_element"(%91) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %94 = "mhlo.get_tuple_element"(%47) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %95 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %96 = "mhlo.custom_call"(%93, %46, %arg32, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %97 = "mhlo.get_tuple_element"(%96) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %98 = "mhlo.reshape"(%97) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %99 = "mhlo.custom_call"(%98) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %100 = "mhlo.custom_call"(%99, %42, %43) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %101 = "mhlo.get_tuple_element"(%100) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %102 = "mhlo.get_tuple_element"(%41) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %103 = "mhlo.get_tuple_element"(%41) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %104 = "mhlo.custom_call"(%101, %102, %103) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %105 = "mhlo.custom_call"(%104, %38, %39) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %106 = "mhlo.get_tuple_element"(%105) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %107 = "mhlo.custom_call"(%106, %37, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %108 = "mhlo.get_tuple_element"(%107) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %109 = "mhlo.get_tuple_element"(%100) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %110 = "mhlo.custom_call"(%109, %37, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %111 = "mhlo.get_tuple_element"(%110) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %112 = "mhlo.get_tuple_element"(%105) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %113 = "mhlo.custom_call"(%112, %37, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %114 = "mhlo.get_tuple_element"(%113) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %115 = call @Unknown5(%92, %108, %111, %114) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %116 = "mhlo.get_tuple_element"(%36) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %117 = "mhlo.get_tuple_element"(%36) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %118 = "mhlo.get_tuple_element"(%36) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %119 = "mhlo.custom_call"(%115, %116, %arg23, %117, %118) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %120 = "mhlo.get_tuple_element"(%119) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %121 = "mhlo.get_tuple_element"(%119) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %122 = "mhlo.get_tuple_element"(%34) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %123 = "mhlo.get_tuple_element"(%34) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %124 = "mhlo.custom_call"(%121, %33, %arg21, %122, %123) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %125 = "mhlo.get_tuple_element"(%124) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %126 = "mhlo.get_tuple_element"(%32) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %127 = "mhlo.get_tuple_element"(%32) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %128 = "mhlo.custom_call"(%125, %31, %arg19, %126, %127) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %129 = "mhlo.get_tuple_element"(%128) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %130 = mhlo.add %120, %129 : tensor<2x128x128xf32>
    %131 = "mhlo.get_tuple_element"(%30) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %132 = "mhlo.get_tuple_element"(%30) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %133 = "mhlo.get_tuple_element"(%30) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %134 = "mhlo.custom_call"(%130, %131, %arg17, %132, %133) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %135 = "mhlo.get_tuple_element"(%134) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %136 = "mhlo.get_tuple_element"(%134) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %137 = "mhlo.get_tuple_element"(%28) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %138 = "mhlo.get_tuple_element"(%28) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %139 = "mhlo.custom_call"(%136, %27, %arg15, %137, %138) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %140 = "mhlo.get_tuple_element"(%139) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %141 = "mhlo.reshape"(%140) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %142 = "mhlo.custom_call"(%141) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %143 = "mhlo.custom_call"(%142, %23, %24) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %144 = "mhlo.get_tuple_element"(%143) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %145 = "mhlo.get_tuple_element"(%22) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %146 = "mhlo.get_tuple_element"(%22) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %147 = "mhlo.custom_call"(%144, %145, %146) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %148 = "mhlo.custom_call"(%147, %19, %20) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %149 = "mhlo.get_tuple_element"(%148) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %150 = "mhlo.custom_call"(%149, %18, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %151 = "mhlo.get_tuple_element"(%150) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %152 = "mhlo.get_tuple_element"(%143) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %153 = "mhlo.custom_call"(%152, %18, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %154 = "mhlo.get_tuple_element"(%153) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %155 = "mhlo.get_tuple_element"(%148) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %156 = "mhlo.custom_call"(%155, %18, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %157 = "mhlo.get_tuple_element"(%156) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %158 = call @Unknown6(%135, %151, %154, %157) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %159 = "mhlo.get_tuple_element"(%17) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %160 = "mhlo.get_tuple_element"(%17) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %161 = "mhlo.get_tuple_element"(%17) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %162 = "mhlo.custom_call"(%158, %159, %arg6, %160, %161) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %163 = "mhlo.get_tuple_element"(%162) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %164:2 = call @Unknown7(%7#2, %163, %11#2) : (tensor<256xi1>, tensor<2x128x128xf32>, tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>)
    %165 = "mhlo.scatter"(%64, %7#1, %164#0) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %166 = "mhlo.scatter"(%0, %11#1, %164#1) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %167 = "mhlo.get_tuple_element"(%162) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %168 = "mhlo.reduce"(%167, %2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %169 = call @Unknown8(%14#2, %168) : (tensor<128xi1>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %170 = "mhlo.scatter"(%1, %14#1, %169) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %171 = "mhlo.get_tuple_element"(%162) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %172 = "mhlo.get_tuple_element"(%162) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %173 = "mhlo.get_tuple_element"(%150) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %174 = "mhlo.get_tuple_element"(%150) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %175 = "mhlo.get_tuple_element"(%156) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %176 = "mhlo.get_tuple_element"(%156) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %177 = "mhlo.get_tuple_element"(%153) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %178 = "mhlo.get_tuple_element"(%153) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %179 = "mhlo.get_tuple_element"(%139) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %180 = "mhlo.get_tuple_element"(%139) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %181 = "mhlo.get_tuple_element"(%134) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %182 = "mhlo.get_tuple_element"(%134) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %183 = "mhlo.get_tuple_element"(%128) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %184 = "mhlo.get_tuple_element"(%128) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %185 = "mhlo.get_tuple_element"(%124) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %186 = "mhlo.get_tuple_element"(%124) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %187 = "mhlo.get_tuple_element"(%119) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %188 = "mhlo.get_tuple_element"(%119) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %189 = "mhlo.get_tuple_element"(%107) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %190 = "mhlo.get_tuple_element"(%107) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %191 = "mhlo.get_tuple_element"(%113) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %192 = "mhlo.get_tuple_element"(%113) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %193 = "mhlo.get_tuple_element"(%110) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %194 = "mhlo.get_tuple_element"(%110) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %195 = "mhlo.get_tuple_element"(%96) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %196 = "mhlo.get_tuple_element"(%96) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %197 = "mhlo.get_tuple_element"(%91) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %198 = "mhlo.get_tuple_element"(%91) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %199 = "mhlo.get_tuple_element"(%85) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %200 = "mhlo.get_tuple_element"(%85) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %201 = "mhlo.get_tuple_element"(%81) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %202 = "mhlo.get_tuple_element"(%81) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %203 = "mhlo.get_tuple_element"(%76) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %204 = "mhlo.get_tuple_element"(%76) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %205 = "mhlo.get_tuple_element"(%71) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %206 = "mhlo.get_tuple_element"(%71) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %207 = "mhlo.get_tuple_element"(%67) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %208 = "mhlo.get_tuple_element"(%67) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %209 = "mhlo.reduce"(%arg47, %2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %210 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%210) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    return %63, %165, %166, %170, %171, %172, %173, %174, %175, %176, %177, %178, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %205, %206, %207, %208, %209 : tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>
  }
}

