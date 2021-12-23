// RUN: byteir-opt %s -hlo-fusion-to-linalg="anchor-tag=byre_elementwise_fusion" -unrealized-cast-to-linalg -linalg-fuse-elementwise-ops -cse | FileCheck %s

// CHECK-LABEL: func @main
module  {
  func private @Unknown0(%arg0: tensor<2x128xi64>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>, %arg3: tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %1 = "mhlo.convert"(%0) : (tensor<256xi64>) -> tensor<256xui32>
    %2 = "mhlo.compare"(%0, %arg1) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %3 = mhlo.add %0, %arg2 : tensor<256xi64>
    %4 = "mhlo.select"(%2, %3, %0) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %5 = "mhlo.reshape"(%4) : (tensor<256xi64>) -> tensor<256x1xi64>
    %6 = "mhlo.convert"(%0) : (tensor<256xi64>) -> tensor<256xf64>
    %7 = "mhlo.compare"(%6, %arg3) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %1, %5, %7 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown1(%arg0: tensor<128xi64>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>, %arg3: tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    %1 = "mhlo.reshape"(%0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %2 = "mhlo.convert"(%1) : (tensor<256xi64>) -> tensor<256xui32>
    %3 = "mhlo.compare"(%1, %arg1) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %4 = mhlo.add %1, %arg2 : tensor<256xi64>
    %5 = "mhlo.select"(%3, %4, %1) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %6 = "mhlo.reshape"(%5) : (tensor<256xi64>) -> tensor<256x1xi64>
    %7 = "mhlo.convert"(%1) : (tensor<256xi64>) -> tensor<256xf64>
    %8 = "mhlo.compare"(%7, %arg3) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %2, %6, %8 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func private @Unknown2(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %1 = "mhlo.reshape"(%arg1) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %2 = mhlo.add %0, %1 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func private @Unknown3(%arg0: tensor<1x128xi64>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>, %arg3: tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "mhlo.compare"(%0, %arg1) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %3 = mhlo.add %0, %arg2 : tensor<128xi64>
    %4 = "mhlo.select"(%2, %3, %0) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %5 = "mhlo.reshape"(%4) : (tensor<128xi64>) -> tensor<128x1xi64>
    %6 = "mhlo.convert"(%0) : (tensor<128xi64>) -> tensor<128xf64>
    %7 = "mhlo.compare"(%6, %arg3) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %1, %5, %7 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func private @Unknown4(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func private @Unknown5(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func private @Unknown6(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256x128xf32>, %arg3: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %1 = "mhlo.reshape"(%arg1) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %2 = "mhlo.select"(%0, %1, %arg2) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %4 = "mhlo.select"(%3, %1, %arg2) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %2, %4 : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func private @Unknown7(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {byre_elementwise_fusion} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
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
    %12 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %14 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %15 = "mhlo.reshape"(%14) : (tensor<1x128xi64>) -> tensor<128xi64>
    %16 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %17:3 = call @Unknown0(%arg0, %4, %1, %2) : (tensor<2x128xi64>, tensor<256xi64>, tensor<256xi64>, tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %18 = "mhlo.gather"(%arg3, %17#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %19:3 = call @Unknown1(%15, %4, %5, %6) : (tensor<128xi64>, tensor<256xi64>, tensor<256xi64>, tensor<256xf64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %20 = "mhlo.gather"(%arg4, %19#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %21 = call @Unknown2(%18, %20) : (tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %22:3 = call @Unknown3(%16, %9, %10, %11) : (tensor<1x128xi64>, tensor<128xi64>, tensor<128xi64>, tensor<128xf64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %23 = "mhlo.gather"(%arg5, %22#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %24 = "mhlo.reshape"(%23) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %25 = "mhlo.custom_call"(%21, %arg6, %arg7, %24) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %26 = "mhlo.get_tuple_element"(%25) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %27 = "mhlo.custom_call"(%26, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %28 = "mhlo.custom_call"(%26, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %29 = "mhlo.custom_call"(%27, %28) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %30 = "mhlo.custom_call"(%29, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %31 = "mhlo.get_tuple_element"(%30) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %32 = "mhlo.custom_call"(%26, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %33 = "mhlo.custom_call"(%31, %32) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %34 = "mhlo.custom_call"(%33) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %35 = "mhlo.reshape"(%34) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %36 = "mhlo.custom_call"(%35, %arg15, %arg16) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %37 = "mhlo.get_tuple_element"(%36) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %38 = "mhlo.custom_call"(%37, %arg17, %arg18, %26) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %39 = "mhlo.get_tuple_element"(%38) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %40 = "mhlo.custom_call"(%39, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %41 = "mhlo.get_tuple_element"(%40) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %42 = "mhlo.custom_call"(%41, %arg21, %arg22) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %44 = "mhlo.custom_call"(%43, %arg23, %arg24, %39) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %45 = "mhlo.get_tuple_element"(%44) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %46 = "mhlo.custom_call"(%45, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %47 = "mhlo.custom_call"(%45, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %48 = "mhlo.custom_call"(%46, %47) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %49 = "mhlo.custom_call"(%48, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %50 = "mhlo.get_tuple_element"(%49) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %51 = "mhlo.custom_call"(%45, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %52 = "mhlo.custom_call"(%50, %51) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %53 = "mhlo.custom_call"(%52) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %54 = "mhlo.reshape"(%53) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %55 = "mhlo.custom_call"(%54, %arg32, %arg33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %56 = "mhlo.get_tuple_element"(%55) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %57 = "mhlo.custom_call"(%56, %arg34, %arg35, %45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %58 = "mhlo.get_tuple_element"(%57) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %59 = "mhlo.custom_call"(%58, %arg36, %arg37) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %60 = "mhlo.get_tuple_element"(%59) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %61 = "mhlo.custom_call"(%60, %arg38, %arg39) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %62 = "mhlo.get_tuple_element"(%61) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %63 = "mhlo.custom_call"(%62, %arg40, %arg41, %58) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %64 = "mhlo.get_tuple_element"(%63) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %65 = "mhlo.custom_call"(%64, %arg42, %arg43) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %66 = "mhlo.get_tuple_element"(%65) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %67 = "mhlo.custom_call"(%66, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %68 = "mhlo.get_tuple_element"(%67) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %69 = "mhlo.custom_call"(%68, %arg3, %arg46) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %70 = "mhlo.custom_call"(%arg47, %68, %arg3) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x30522xf32>, tensor<2x128x128xf32>, tensor<30522x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    %71 = "mhlo.get_tuple_element"(%70) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<30522x128xf32>
    %72 = "mhlo.get_tuple_element"(%70) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<2x128x128xf32>
    %73 = "mhlo.get_tuple_element"(%67) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %74 = "mhlo.get_tuple_element"(%67) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %75 = "mhlo.custom_call"(%72, %66, %arg44, %73, %74) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %76 = "mhlo.get_tuple_element"(%75) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %77 = "mhlo.get_tuple_element"(%65) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %78 = "mhlo.get_tuple_element"(%65) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %79 = "mhlo.custom_call"(%76, %64, %arg42, %77, %78) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %80 = "mhlo.get_tuple_element"(%79) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %81 = "mhlo.get_tuple_element"(%63) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %82 = "mhlo.get_tuple_element"(%63) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %83 = "mhlo.get_tuple_element"(%63) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %84 = "mhlo.custom_call"(%80, %81, %arg40, %82, %83) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %85 = "mhlo.get_tuple_element"(%84) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %86 = "mhlo.get_tuple_element"(%84) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %87 = "mhlo.get_tuple_element"(%61) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %88 = "mhlo.get_tuple_element"(%61) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %89 = "mhlo.custom_call"(%86, %60, %arg38, %87, %88) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %90 = "mhlo.get_tuple_element"(%89) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %91 = "mhlo.get_tuple_element"(%59) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %92 = "mhlo.get_tuple_element"(%59) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %93 = "mhlo.custom_call"(%90, %58, %arg36, %91, %92) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %94 = "mhlo.get_tuple_element"(%93) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %95 = mhlo.add %85, %94 : tensor<2x128x128xf32>
    %96 = "mhlo.get_tuple_element"(%57) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %97 = "mhlo.get_tuple_element"(%57) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %98 = "mhlo.get_tuple_element"(%57) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %99 = "mhlo.custom_call"(%95, %96, %arg34, %97, %98) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %100 = "mhlo.get_tuple_element"(%99) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %101 = "mhlo.get_tuple_element"(%99) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %102 = "mhlo.get_tuple_element"(%55) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %103 = "mhlo.get_tuple_element"(%55) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %104 = "mhlo.custom_call"(%101, %54, %arg32, %102, %103) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %105 = "mhlo.get_tuple_element"(%104) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %106 = "mhlo.reshape"(%105) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %107 = "mhlo.custom_call"(%106) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %108 = "mhlo.custom_call"(%107, %50, %51) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %109 = "mhlo.get_tuple_element"(%108) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %110 = "mhlo.get_tuple_element"(%49) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %111 = "mhlo.get_tuple_element"(%49) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %112 = "mhlo.custom_call"(%109, %110, %111) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %113 = "mhlo.custom_call"(%112, %46, %47) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %114 = "mhlo.get_tuple_element"(%113) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %115 = "mhlo.custom_call"(%114, %45, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %116 = "mhlo.get_tuple_element"(%115) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %117 = "mhlo.get_tuple_element"(%108) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %118 = "mhlo.custom_call"(%117, %45, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %120 = "mhlo.get_tuple_element"(%113) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %121 = "mhlo.custom_call"(%120, %45, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %122 = "mhlo.get_tuple_element"(%121) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %123 = call @Unknown4(%100, %116, %119, %122) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %124 = "mhlo.get_tuple_element"(%44) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %125 = "mhlo.get_tuple_element"(%44) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %126 = "mhlo.get_tuple_element"(%44) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %127 = "mhlo.custom_call"(%123, %124, %arg23, %125, %126) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %128 = "mhlo.get_tuple_element"(%127) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %129 = "mhlo.get_tuple_element"(%127) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %130 = "mhlo.get_tuple_element"(%42) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %131 = "mhlo.get_tuple_element"(%42) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %132 = "mhlo.custom_call"(%129, %41, %arg21, %130, %131) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %134 = "mhlo.get_tuple_element"(%40) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %135 = "mhlo.get_tuple_element"(%40) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %136 = "mhlo.custom_call"(%133, %39, %arg19, %134, %135) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %137 = "mhlo.get_tuple_element"(%136) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %138 = mhlo.add %128, %137 : tensor<2x128x128xf32>
    %139 = "mhlo.get_tuple_element"(%38) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %140 = "mhlo.get_tuple_element"(%38) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %141 = "mhlo.get_tuple_element"(%38) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %142 = "mhlo.custom_call"(%138, %139, %arg17, %140, %141) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %143 = "mhlo.get_tuple_element"(%142) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %144 = "mhlo.get_tuple_element"(%142) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %145 = "mhlo.get_tuple_element"(%36) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %146 = "mhlo.get_tuple_element"(%36) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %147 = "mhlo.custom_call"(%144, %35, %arg15, %145, %146) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %148 = "mhlo.get_tuple_element"(%147) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %149 = "mhlo.reshape"(%148) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %150 = "mhlo.custom_call"(%149) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %151 = "mhlo.custom_call"(%150, %31, %32) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %152 = "mhlo.get_tuple_element"(%151) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %153 = "mhlo.get_tuple_element"(%30) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %154 = "mhlo.get_tuple_element"(%30) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %155 = "mhlo.custom_call"(%152, %153, %154) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %156 = "mhlo.custom_call"(%155, %27, %28) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %157 = "mhlo.get_tuple_element"(%156) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %158 = "mhlo.custom_call"(%157, %26, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %159 = "mhlo.get_tuple_element"(%158) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %160 = "mhlo.get_tuple_element"(%151) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %161 = "mhlo.custom_call"(%160, %26, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %162 = "mhlo.get_tuple_element"(%161) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %163 = "mhlo.get_tuple_element"(%156) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %164 = "mhlo.custom_call"(%163, %26, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %165 = "mhlo.get_tuple_element"(%164) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %166 = call @Unknown5(%143, %159, %162, %165) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %167 = "mhlo.get_tuple_element"(%25) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %168 = "mhlo.get_tuple_element"(%25) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %169 = "mhlo.get_tuple_element"(%25) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %170 = "mhlo.custom_call"(%166, %167, %arg6, %168, %169) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %171 = "mhlo.get_tuple_element"(%170) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %172:2 = call @Unknown6(%17#2, %171, %7, %19#2) : (tensor<256xi1>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>)
    %173 = "mhlo.scatter"(%0, %17#1, %172#0) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %219 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%219) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %174 = mhlo.add %71, %173 : tensor<30522x128xf32>
    %175 = "mhlo.scatter"(%3, %19#1, %172#1) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %219 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%219) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %176 = "mhlo.get_tuple_element"(%170) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %177 = "mhlo.reduce"(%176, %12) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %219 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%219) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %178 = call @Unknown7(%22#2, %177, %13) : (tensor<128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %179 = "mhlo.scatter"(%8, %22#1, %178) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %219 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%219) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %180 = "mhlo.get_tuple_element"(%170) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %181 = "mhlo.get_tuple_element"(%170) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %182 = "mhlo.get_tuple_element"(%158) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %183 = "mhlo.get_tuple_element"(%158) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %184 = "mhlo.get_tuple_element"(%164) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %185 = "mhlo.get_tuple_element"(%164) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %186 = "mhlo.get_tuple_element"(%161) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %187 = "mhlo.get_tuple_element"(%161) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %188 = "mhlo.get_tuple_element"(%147) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %189 = "mhlo.get_tuple_element"(%147) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %190 = "mhlo.get_tuple_element"(%142) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %191 = "mhlo.get_tuple_element"(%142) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %192 = "mhlo.get_tuple_element"(%136) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %193 = "mhlo.get_tuple_element"(%136) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %194 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %195 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %196 = "mhlo.get_tuple_element"(%127) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %197 = "mhlo.get_tuple_element"(%127) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %198 = "mhlo.get_tuple_element"(%115) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %199 = "mhlo.get_tuple_element"(%115) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %200 = "mhlo.get_tuple_element"(%121) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %201 = "mhlo.get_tuple_element"(%121) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %202 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %203 = "mhlo.get_tuple_element"(%118) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %204 = "mhlo.get_tuple_element"(%104) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %205 = "mhlo.get_tuple_element"(%104) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %206 = "mhlo.get_tuple_element"(%99) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %207 = "mhlo.get_tuple_element"(%99) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %208 = "mhlo.get_tuple_element"(%93) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %209 = "mhlo.get_tuple_element"(%93) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %210 = "mhlo.get_tuple_element"(%89) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %211 = "mhlo.get_tuple_element"(%89) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %212 = "mhlo.get_tuple_element"(%84) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %213 = "mhlo.get_tuple_element"(%84) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %214 = "mhlo.get_tuple_element"(%79) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %215 = "mhlo.get_tuple_element"(%79) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %216 = "mhlo.get_tuple_element"(%75) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %217 = "mhlo.get_tuple_element"(%75) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %218 = "mhlo.get_tuple_element"(%70) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<30522xf32>
    return %69, %174, %175, %179, %180, %181, %182, %183, %184, %185, %186, %187, %188, %189, %190, %191, %192, %193, %194, %195, %196, %197, %198, %199, %200, %201, %202, %203, %204, %205, %206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %216, %217, %218 : tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>
  }
}

