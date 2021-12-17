// RUN: byteir-opt %s --convert-hlo-to-lhlo -cse --linalg-bufferize --cse --canonicalize | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func private @MatmulOp0(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @MatmulOp1(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @Unknown0(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %1 = linalg.init_tensor [128] : tensor<128xui32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%1 : tensor<128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %8 = trunci %arg4 : i64 to i32
      %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
      linalg.yield %9 : ui32
    } -> tensor<128xui32>
    %3 = linalg.init_tensor [128] : tensor<128xi64>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg1, %arg2 : tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) outs(%3 : tensor<128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %8 = addi %arg4, %arg6 : i64
      %9 = cmpi slt, %arg4, %arg5 : i64
      %10 = select %9, %8, %arg4 : i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %5 = linalg.tensor_expand_shape %4 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %6 = linalg.init_tensor [128] : tensor<128xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg3 : tensor<128xi64>, tensor<128xf64>) outs(%6 : tensor<128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %8 = sitofp %arg4 : i64 to f64
      %9 = cmpf une, %8, %arg5 : f64
      linalg.yield %9 : i1
    } -> tensor<128xi1>
    return %2, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown1(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %1 = linalg.init_tensor [128] : tensor<128xui32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%1 : tensor<128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %8 = trunci %arg4 : i64 to i32
      %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
      linalg.yield %9 : ui32
    } -> tensor<128xui32>
    %3 = linalg.init_tensor [128] : tensor<128xi64>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg1, %arg2 : tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) outs(%3 : tensor<128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %8 = addi %arg4, %arg6 : i64
      %9 = cmpi slt, %arg4, %arg5 : i64
      %10 = select %9, %8, %arg4 : i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %5 = linalg.tensor_expand_shape %4 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %6 = linalg.init_tensor [128] : tensor<128xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg3 : tensor<128xi64>, tensor<128xf64>) outs(%6 : tensor<128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %8 = sitofp %arg4 : i64 to f64
      %9 = cmpf une, %8, %arg5 : f64
      linalg.yield %9 : i1
    } -> tensor<128xi1>
    return %2, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown2(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %1 = linalg.init_tensor [128] : tensor<128xui32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%0 : tensor<128xi64>) outs(%1 : tensor<128xui32>) {
    ^bb0(%arg4: i64, %arg5: ui32):  // no predecessors
      %8 = trunci %arg4 : i64 to i32
      %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
      linalg.yield %9 : ui32
    } -> tensor<128xui32>
    %3 = linalg.init_tensor [128] : tensor<128xi64>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg1, %arg2 : tensor<128xi64>, tensor<128xi64>, tensor<128xi64>) outs(%3 : tensor<128xi64>) {
    ^bb0(%arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64):  // no predecessors
      %8 = addi %arg4, %arg6 : i64
      %9 = cmpi slt, %arg4, %arg5 : i64
      %10 = select %9, %8, %arg4 : i64
      linalg.yield %10 : i64
    } -> tensor<128xi64>
    %5 = linalg.tensor_expand_shape %4 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    %6 = linalg.init_tensor [128] : tensor<128xi1>
    %7 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%0, %arg3 : tensor<128xi64>, tensor<128xf64>) outs(%6 : tensor<128xi1>) {
    ^bb0(%arg4: i64, %arg5: f64, %arg6: i1):  // no predecessors
      %8 = sitofp %arg4 : i64 to f64
      %9 = cmpf une, %8, %arg5 : f64
      linalg.yield %9 : i1
    } -> tensor<128xi1>
    return %2, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown3(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<128x128xf32> into tensor<1x128x128xf32>
    %1 = linalg.tensor_expand_shape %arg1 [[0, 1], [2]] : tensor<128x128xf32> into tensor<1x128x128xf32>
    %2 = linalg.tensor_expand_shape %arg2 [[0, 1], [2]] : tensor<128x128xf32> into tensor<1x128x128xf32>
    %3 = linalg.init_tensor [1, 128, 128] : tensor<1x128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1, %2 : tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%3 : tensor<1x128x128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %6 = addf %arg3, %arg4 : f32
      %7 = addf %6, %arg5 : f32
      linalg.yield %7 : f32
    } -> tensor<1x128x128xf32>
    %5 = linalg.tensor_collapse_shape %4 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    return %5 : tensor<128x128xf32>
  }
  func private @Unknown4(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x128x128xf32>, %arg4: tensor<1x128x128xf32>, %arg5: tensor<1x128x128xf32>, %arg6: tensor<1x128x128xf32>, %arg7: tensor<1x128x128xf32>, %arg8: tensor<1x128x128xf32>, %arg9: tensor<1x128x128xf32>, %arg10: tensor<1x128x128xf32>, %arg11: tensor<1x128x128xf32>, %arg12: tensor<1x128x128xf32>, %arg13: tensor<1x128x128xf32>, %arg14: tensor<1x128x128xf32>, %arg15: tensor<1x128x128xf32>, %arg16: tensor<1x128x128xf32>, %arg17: tensor<1x128x128xf32>, %arg18: tensor<1x128x128xf32>, %arg19: tensor<1x128x128xf32>, %arg20: tensor<1x128x128xf32>, %arg21: tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<128x128xf32> into tensor<1x128x128xf32>
    %1 = linalg.init_tensor [1, 128, 128] : tensor<1x128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1, %arg2, %arg4, %arg3, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19 : tensor<1x128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%1 : tensor<1x128x128xf32>) {
    ^bb0(%arg22: f32, %arg23: f32, %arg24: f32, %arg25: f32, %arg26: f32, %arg27: f32, %arg28: f32, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: f32, %arg33: f32, %arg34: f32, %arg35: f32, %arg36: f32, %arg37: f32, %arg38: f32, %arg39: f32, %arg40: f32, %arg41: f32, %arg42: f32):  // no predecessors
      %4 = addf %arg22, %arg23 : f32
      %5 = mulf %4, %arg26 : f32
      %6 = minf %5, %arg27 : f32
      %7 = maxf %6, %arg25 : f32
      %8 = mulf %7, %7 : f32
      %9 = mulf %8, %arg28 : f32
      %10 = addf %9, %arg36 : f32
      %11 = mulf %10, %8 : f32
      %12 = addf %11, %arg37 : f32
      %13 = mulf %12, %8 : f32
      %14 = addf %13, %arg38 : f32
      %15 = mulf %14, %8 : f32
      %16 = addf %15, %arg39 : f32
      %17 = mulf %16, %8 : f32
      %18 = addf %17, %arg40 : f32
      %19 = addf %9, %arg29 : f32
      %20 = mulf %19, %8 : f32
      %21 = addf %20, %arg30 : f32
      %22 = mulf %21, %8 : f32
      %23 = addf %22, %arg31 : f32
      %24 = mulf %23, %8 : f32
      %25 = addf %24, %arg32 : f32
      %26 = mulf %25, %8 : f32
      %27 = addf %26, %arg33 : f32
      %28 = mulf %27, %8 : f32
      %29 = addf %28, %arg34 : f32
      %30 = mulf %29, %8 : f32
      %31 = addf %30, %arg35 : f32
      %32 = mulf %7, %31 : f32
      %33 = divf %32, %18 : f32
      %34 = addf %33, %arg41 : f32
      %35 = mulf %4, %arg24 : f32
      %36 = mulf %35, %34 : f32
      linalg.yield %36 : f32
    } -> tensor<1x128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg4, %0, %arg1, %arg3, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg2, %arg20, %arg21 : tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%1 : tensor<1x128x128xf32>) {
    ^bb0(%arg22: f32, %arg23: f32, %arg24: f32, %arg25: f32, %arg26: f32, %arg27: f32, %arg28: f32, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: f32, %arg33: f32, %arg34: f32, %arg35: f32, %arg36: f32, %arg37: f32, %arg38: f32, %arg39: f32, %arg40: f32, %arg41: f32, %arg42: f32, %arg43: f32, %arg44: f32):  // no predecessors
      %4 = addf %arg23, %arg24 : f32
      %5 = mulf %4, %4 : f32
      %6 = mulf %5, %arg42 : f32
      %7 = math.exp %6 : f32
      %8 = mulf %4, %7 : f32
      %9 = mulf %8, %arg43 : f32
      %10 = mulf %4, %arg25 : f32
      %11 = minf %10, %arg26 : f32
      %12 = maxf %11, %arg22 : f32
      %13 = mulf %12, %12 : f32
      %14 = mulf %13, %arg27 : f32
      %15 = addf %14, %arg35 : f32
      %16 = mulf %15, %13 : f32
      %17 = addf %16, %arg36 : f32
      %18 = mulf %17, %13 : f32
      %19 = addf %18, %arg37 : f32
      %20 = mulf %19, %13 : f32
      %21 = addf %20, %arg38 : f32
      %22 = mulf %21, %13 : f32
      %23 = addf %22, %arg39 : f32
      %24 = addf %14, %arg28 : f32
      %25 = mulf %24, %13 : f32
      %26 = addf %25, %arg29 : f32
      %27 = mulf %26, %13 : f32
      %28 = addf %27, %arg30 : f32
      %29 = mulf %28, %13 : f32
      %30 = addf %29, %arg31 : f32
      %31 = mulf %30, %13 : f32
      %32 = addf %31, %arg32 : f32
      %33 = mulf %32, %13 : f32
      %34 = addf %33, %arg33 : f32
      %35 = mulf %34, %13 : f32
      %36 = addf %35, %arg34 : f32
      %37 = mulf %12, %36 : f32
      %38 = divf %37, %23 : f32
      %39 = addf %38, %arg40 : f32
      %40 = mulf %39, %arg41 : f32
      %41 = addf %40, %9 : f32
      linalg.yield %41 : f32
    } -> tensor<1x128x128xf32>
    return %2, %3 : tensor<1x128x128xf32>, tensor<1x128x128xf32>
  }
  func private @Unknown5(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [128] : tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg2, %arg0, %arg1 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = addf %arg4, %arg5 : f32
      %3 = math.rsqrt %2 : f32
      %4 = divf %arg3, %3 : f32
      %5 = mulf %4, %4 : f32
      %6 = subf %5, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
  func private @Unknown6(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<128x128xf32> into tensor<1x128x128xf32>
    %1 = linalg.init_tensor [1, 128, 128] : tensor<1x128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : tensor<1x128x128xf32>, tensor<128xf32>) outs(%1 : tensor<1x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %6 = mulf %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<1x128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %arg2, %arg1 : tensor<128xf32>, tensor<1x128x128xf32>, tensor<128xf32>) outs(%1 : tensor<1x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):  // no predecessors
      %6 = mulf %arg5, %arg6 : f32
      %7 = addf %arg4, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<1x128x128xf32>
    %4 = linalg.tensor_collapse_shape %3 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg2 : tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%1 : tensor<1x128x128xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %6 = mulf %arg4, %arg5 : f32
      linalg.yield %6 : f32
    } -> tensor<1x128x128xf32>
    return %0, %2, %4, %5 : tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>
  }
  func private @Unknown7(%arg0: tensor<128x30522xf32>, %arg1: tensor<30522xf32>) -> tensor<1x128x30522xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2]] : tensor<128x30522xf32> into tensor<1x128x30522xf32>
    %1 = linalg.init_tensor [1, 128, 30522] : tensor<1x128x30522xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg1 : tensor<1x128x30522xf32>, tensor<30522xf32>) outs(%1 : tensor<1x128x30522xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = addf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    } -> tensor<1x128x30522xf32>
    return %2 : tensor<1x128x30522xf32>
  }
  func private @Unknown8(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [1, 128, 128] : tensor<1x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x128x128xf32>, tensor<1x128x128xf32>) outs(%0 : tensor<1x128x128xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    } -> tensor<1x128x128xf32>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1], [2]] : tensor<1x128x128xf32> into tensor<128x128xf32>
    return %1, %2 : tensor<1x128x128xf32>, tensor<128x128xf32>
  }
  func private @Unknown9(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [128, 128] : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @Unknown10(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [128, 128] : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @Unknown11(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = linalg.init_tensor [128, 128] : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func @main(%arg0: tensor<30522x128xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<2x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<512x128xf32>, %arg16: tensor<512xf32>, %arg17: tensor<128x512xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128x128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<512x128xf32>, %arg32: tensor<512xf32>, %arg33: tensor<128x512xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<30522xf32>, %arg38: tensor<128x128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<30522x128xf32>, %arg43: tensor<30522xf32>, %arg44: tensor<1x512xi64>, %arg45: tensor<1x512xi64>, %arg46: tensor<1x128xi64>) -> (tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<0.398942292> : tensor<1x128x128xf32>
    %1 = mhlo.constant dense<-5.000000e-01> : tensor<1x128x128xf32>
    %2 = mhlo.constant dense<0.707106769> : tensor<1x128x128xf32>
    %3 = mhlo.constant dense<5.000000e-01> : tensor<1x128x128xf32>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<1x128x30522xf32>
    %6 = mhlo.constant dense<1.000000e+00> : tensor<128x30522xf32>
    %7 = mhlo.constant dense<9.99999996E-13> : tensor<128xf32>
    %8 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %9 = mhlo.constant dense<0.000000e+00> : tensor<30522x128xf32>
    %10 = mhlo.constant dense<30522> : tensor<128xi64>
    %11 = mhlo.constant dense<0.000000e+00> : tensor<128xf64>
    %12 = mhlo.constant dense<-0.0142647391> : tensor<1x128x128xf32>
    %13 = mhlo.constant dense<-0.00737332925> : tensor<1x128x128xf32>
    %14 = mhlo.constant dense<-0.00168282702> : tensor<1x128x128xf32>
    %15 = mhlo.constant dense<-2.13374049E-4> : tensor<1x128x128xf32>
    %16 = mhlo.constant dense<-1.45660715E-5> : tensor<1x128x128xf32>
    %17 = mhlo.constant dense<0.000000e+00> : tensor<1x128x128xf32>
    %18 = mhlo.constant dense<-0.0160960332> : tensor<1x128x128xf32>
    %19 = mhlo.constant dense<-2.954600e-03> : tensor<1x128x128xf32>
    %20 = mhlo.constant dense<-7.34990637E-4> : tensor<1x128x128xf32>
    %21 = mhlo.constant dense<-5.69250624E-5> : tensor<1x128x128xf32>
    %22 = mhlo.constant dense<-2.10102394E-6> : tensor<1x128x128xf32>
    %23 = mhlo.constant dense<2.77068146E-8> : tensor<1x128x128xf32>
    %24 = mhlo.constant dense<-2.72614237E-10> : tensor<1x128x128xf32>
    %25 = mhlo.constant dense<4.000000e+00> : tensor<1x128x128xf32>
    %26 = mhlo.constant dense<-4.000000e+00> : tensor<1x128x128xf32>
    %27 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %28 = mhlo.constant dense<512> : tensor<128xi64>
    %29 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %30 = mhlo.constant dense<0> : tensor<128xi64>
    %31 = mhlo.constant dense<2> : tensor<128xi64>
    %32 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %33 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %34 = mhlo.constant dense<1.000000e+00> : tensor<1x128x128xf32>
    %35 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = "mhlo.slice"(%arg45) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %37 = "mhlo.slice"(%arg44) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %38 = "mhlo.dot"(%6, %arg42) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x30522xf32>, tensor<30522x128xf32>) -> tensor<128x128xf32>
    %39:3 = call @Unknown0(%arg46, %30, %10, %11) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %40 = "mhlo.gather"(%arg0, %39#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %41:3 = call @Unknown1(%37, %30, %28, %32) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %42 = "mhlo.gather"(%arg1, %41#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %43:3 = call @Unknown2(%36, %30, %31, %32) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %44 = "mhlo.gather"(%arg2, %43#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %45 = call @Unknown3(%40, %44, %42) : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %46 = "mhlo.dot_general"(%45, %arg38) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %47:2 = call @Unknown4(%46, %arg39, %3, %2, %26, %25, %17, %24, %23, %22, %21, %20, %19, %18, %16, %15, %14, %13, %12, %34, %1, %0) : (tensor<128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>)
    %48 = "mhlo.batch_norm_training"(%47#0, %8, %4) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %50 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %51 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %52 = call @Unknown5(%51, %7, %8) : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
    %53:4 = call @Unknown6(%38, %arg40, %49, %arg41) : (tensor<128x128xf32>, tensor<128xf32>, tensor<1x128x128xf32>, tensor<128xf32>) -> (tensor<1x128x128xf32>, tensor<1x128x128xf32>, tensor<128x128xf32>, tensor<1x128x128xf32>)
    %54 = "mhlo.batch_norm_grad"(%47#0, %8, %50, %52, %53#1) {epsilon = 9.99999996E-13 : f32, feature_index = 1 : i64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %55 = "mhlo.dot_general"(%53#2, %arg42) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %56 = call @Unknown7(%55, %arg43) : (tensor<128x30522xf32>, tensor<30522xf32>) -> tensor<1x128x30522xf32>
    %57 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<1x128x128xf32>
    %58:2 = call @Unknown8(%57, %47#1) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>)
    %59 = "mhlo.dot"(%58#1, %arg38) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %60 = call @Unknown9(%39#2, %59, %33) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %61 = "mhlo.scatter"(%9, %39#1, %60) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<30522x128xf32>
    %62 = call @Unknown10(%41#2, %59, %33) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %63 = "mhlo.scatter"(%27, %41#1, %62) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %64 = call @Unknown11(%43#2, %59, %33) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %65 = "mhlo.scatter"(%29, %43#1, %64) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %66 = call @MatmulOp0(%45, %58#1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %67 = "mhlo.reduce"(%58#0, %35) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %68 = "mhlo.reduce"(%53#3, %35) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %69 = "mhlo.reduce"(%53#0, %35) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
    %70 = call @MatmulOp1(%53#2, %6) : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %71 = "mhlo.reduce"(%5, %35) ( {
    ^bb0(%arg47: tensor<f32>, %arg48: tensor<f32>):  // no predecessors
      %72 = mhlo.add %arg47, %arg48 : tensor<f32>
      "mhlo.return"(%72) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    return %56, %61, %63, %65, %66, %67, %68, %69, %70, %71 : tensor<1x128x30522xf32>, tensor<30522x128xf32>, tensor<512x128xf32>, tensor<2x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522x128xf32>, tensor<30522xf32>
  }
}

// CHECK-LABEL: func private @MatmulOp0(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @MatmulOp1(%arg0: memref<128x128xf32>, %arg1: memref<128x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}

// CHECK-LABEL: func private @Unknown0

// CHECK-LABEL: func private @Unknown1

// CHECK-LABEL: func @main
