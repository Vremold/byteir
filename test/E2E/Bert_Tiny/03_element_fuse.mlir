// RUN: byteir-opt %s -expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse -fuse-element="attach-tag=byre_elementwise_fusion" -fusion-outlining -cse | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>>
module  {
  func private @MatmulOp0(%arg0: tensor<256x128xf32>, %arg1: tensor<256x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<1x512xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<30522x128xf32>, %arg4: tensor<2x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<2x1x1x128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<2x1x1x128xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<512x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<30522xf32>, %arg47: tensor<2x128x30522xf32>) -> !tuple {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %2 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %3 = mhlo.constant dense<512> : tensor<128xi64>
    %4 = mhlo.constant dense<0> : tensor<128xi64>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %6 = mhlo.constant dense<0.000000e+00> : tensor<256x128xf32>
    %7 = mhlo.constant dense<-1.000000e+00> : tensor<256xf64>
    %8 = mhlo.constant dense<2> : tensor<256xi64>
    %9 = mhlo.constant dense<0> : tensor<256xi64>
    %10 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %11 = mhlo.constant dense<0.000000e+00> : tensor<256xf64>
    %12 = mhlo.constant dense<30522> : tensor<256xi64>
    %13 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %14 = "mhlo.convert"(%13) : (tensor<256xi64>) -> tensor<256xui32>
    %15 = "mhlo.gather"(%arg3, %14) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %16 = "mhlo.reshape"(%15) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %17 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %18 = "mhlo.reshape"(%17) : (tensor<1x128xi64>) -> tensor<128xi64>
    %19 = "mhlo.broadcast_in_dim"(%18) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    %20 = "mhlo.reshape"(%19) : (tensor<2x128xi64>) -> tensor<256xi64>
    %21 = "mhlo.convert"(%20) : (tensor<256xi64>) -> tensor<256xui32>
    %22 = "mhlo.gather"(%arg4, %21) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %23 = "mhlo.reshape"(%22) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %24 = mhlo.add %16, %23 : tensor<2x128x128xf32>
    %25 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %26 = "mhlo.reshape"(%25) : (tensor<1x128xi64>) -> tensor<128xi64>
    %27 = "mhlo.convert"(%26) : (tensor<128xi64>) -> tensor<128xui32>
    %28 = "mhlo.gather"(%arg5, %27) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %29 = "mhlo.reshape"(%28) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %30 = "mhlo.custom_call"(%24, %arg6, %arg7, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %31 = "mhlo.get_tuple_element"(%30) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %32 = "mhlo.custom_call"(%31, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %33 = "mhlo.custom_call"(%31, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %34 = "mhlo.custom_call"(%32, %33) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %35 = "mhlo.custom_call"(%34, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %36 = "mhlo.get_tuple_element"(%35) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %37 = "mhlo.custom_call"(%31, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %38 = "mhlo.custom_call"(%36, %37) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %39 = "mhlo.custom_call"(%38) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %40 = "mhlo.reshape"(%39) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %41 = "mhlo.custom_call"(%40, %arg15, %arg16) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %42 = "mhlo.get_tuple_element"(%41) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %43 = "mhlo.custom_call"(%42, %arg17, %arg18, %31) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %44 = "mhlo.get_tuple_element"(%43) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %45 = "mhlo.custom_call"(%44, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %46 = "mhlo.get_tuple_element"(%45) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %47 = "mhlo.custom_call"(%46, %arg21, %arg22) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %49 = "mhlo.custom_call"(%48, %arg23, %arg24, %44) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %50 = "mhlo.get_tuple_element"(%49) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %51 = "mhlo.custom_call"(%50, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %52 = "mhlo.custom_call"(%50, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %53 = "mhlo.custom_call"(%51, %52) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %54 = "mhlo.custom_call"(%53, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %55 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %56 = "mhlo.custom_call"(%50, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %57 = "mhlo.custom_call"(%55, %56) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %58 = "mhlo.custom_call"(%57) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %59 = "mhlo.reshape"(%58) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %60 = "mhlo.custom_call"(%59, %arg32, %arg33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %61 = "mhlo.get_tuple_element"(%60) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %62 = "mhlo.custom_call"(%61, %arg34, %arg35, %50) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %63 = "mhlo.get_tuple_element"(%62) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %64 = "mhlo.custom_call"(%63, %arg36, %arg37) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %65 = "mhlo.get_tuple_element"(%64) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %66 = "mhlo.custom_call"(%65, %arg38, %arg39) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %67 = "mhlo.get_tuple_element"(%66) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %68 = "mhlo.custom_call"(%67, %arg40, %arg41, %63) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %69 = "mhlo.get_tuple_element"(%68) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %70 = "mhlo.custom_call"(%69, %arg42, %arg43) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %71 = "mhlo.get_tuple_element"(%70) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %72 = "mhlo.custom_call"(%71, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %73 = "mhlo.get_tuple_element"(%72) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %74 = "mhlo.reshape"(%73) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %75 = "mhlo.dot_general"(%74, %arg3) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<30522x128xf32>) -> tensor<256x30522xf32>
    %76 = "mhlo.reshape"(%75) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %77 = "mhlo.broadcast_in_dim"(%arg46) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %78 = mhlo.add %76, %77 : tensor<2x128x30522xf32>
    %79 = "mhlo.reshape"(%arg47) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %80 = call @MatmulOp0(%74, %79) : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<30522x128xf32>
    %81 = "mhlo.compare"(%13, %9) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %82 = mhlo.add %13, %12 : tensor<256xi64>
    %83 = "mhlo.select"(%81, %82, %13) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %84 = "mhlo.reshape"(%83) : (tensor<256xi64>) -> tensor<256x1xi64>
    %85 = "mhlo.convert"(%13) : (tensor<256xi64>) -> tensor<256xf64>
    %86 = "mhlo.compare"(%85, %11) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    %87 = "mhlo.broadcast_in_dim"(%86) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %88 = "mhlo.dot"(%79, %arg3) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %89 = "mhlo.reshape"(%88) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %90 = "mhlo.get_tuple_element"(%72) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %91 = "mhlo.get_tuple_element"(%72) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %92 = "mhlo.custom_call"(%89, %71, %arg44, %90, %91) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %93 = "mhlo.get_tuple_element"(%92) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %94 = "mhlo.get_tuple_element"(%70) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %95 = "mhlo.get_tuple_element"(%70) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %96 = "mhlo.custom_call"(%93, %69, %arg42, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %97 = "mhlo.get_tuple_element"(%96) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %98 = "mhlo.get_tuple_element"(%68) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %99 = "mhlo.get_tuple_element"(%68) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %100 = "mhlo.get_tuple_element"(%68) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %101 = "mhlo.custom_call"(%97, %98, %arg40, %99, %100) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %102 = "mhlo.get_tuple_element"(%101) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %103 = "mhlo.get_tuple_element"(%101) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %104 = "mhlo.get_tuple_element"(%66) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %105 = "mhlo.get_tuple_element"(%66) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %106 = "mhlo.custom_call"(%103, %65, %arg38, %104, %105) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %107 = "mhlo.get_tuple_element"(%106) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %108 = "mhlo.get_tuple_element"(%64) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %109 = "mhlo.get_tuple_element"(%64) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %110 = "mhlo.custom_call"(%107, %63, %arg36, %108, %109) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %111 = "mhlo.get_tuple_element"(%110) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %112 = mhlo.add %102, %111 : tensor<2x128x128xf32>
    %113 = "mhlo.get_tuple_element"(%62) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %114 = "mhlo.get_tuple_element"(%62) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %115 = "mhlo.get_tuple_element"(%62) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %116 = "mhlo.custom_call"(%112, %113, %arg34, %114, %115) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %117 = "mhlo.get_tuple_element"(%116) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %118 = "mhlo.get_tuple_element"(%116) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %119 = "mhlo.get_tuple_element"(%60) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %120 = "mhlo.get_tuple_element"(%60) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %121 = "mhlo.custom_call"(%118, %59, %arg32, %119, %120) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %122 = "mhlo.get_tuple_element"(%121) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %123 = "mhlo.reshape"(%122) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %124 = "mhlo.custom_call"(%123) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %125 = "mhlo.custom_call"(%124, %55, %56) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %126 = "mhlo.get_tuple_element"(%125) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %127 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %128 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %129 = "mhlo.custom_call"(%126, %127, %128) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %130 = "mhlo.custom_call"(%129, %51, %52) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %131 = "mhlo.get_tuple_element"(%130) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %132 = "mhlo.custom_call"(%131, %50, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %134 = mhlo.add %117, %133 : tensor<2x128x128xf32>
    %135 = "mhlo.get_tuple_element"(%125) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %136 = "mhlo.custom_call"(%135, %50, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %137 = "mhlo.get_tuple_element"(%136) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %138 = mhlo.add %134, %137 : tensor<2x128x128xf32>
    %139 = "mhlo.get_tuple_element"(%130) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %140 = "mhlo.custom_call"(%139, %50, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %141 = "mhlo.get_tuple_element"(%140) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %142 = mhlo.add %138, %141 : tensor<2x128x128xf32>
    %143 = "mhlo.get_tuple_element"(%49) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %144 = "mhlo.get_tuple_element"(%49) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %145 = "mhlo.get_tuple_element"(%49) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %146 = "mhlo.custom_call"(%142, %143, %arg23, %144, %145) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %147 = "mhlo.get_tuple_element"(%146) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %148 = "mhlo.get_tuple_element"(%146) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %149 = "mhlo.get_tuple_element"(%47) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %150 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %151 = "mhlo.custom_call"(%148, %46, %arg21, %149, %150) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %152 = "mhlo.get_tuple_element"(%151) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %153 = "mhlo.get_tuple_element"(%45) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %154 = "mhlo.get_tuple_element"(%45) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %155 = "mhlo.custom_call"(%152, %44, %arg19, %153, %154) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %156 = "mhlo.get_tuple_element"(%155) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %157 = mhlo.add %147, %156 : tensor<2x128x128xf32>
    %158 = "mhlo.get_tuple_element"(%43) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %159 = "mhlo.get_tuple_element"(%43) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %160 = "mhlo.get_tuple_element"(%43) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %161 = "mhlo.custom_call"(%157, %158, %arg17, %159, %160) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %162 = "mhlo.get_tuple_element"(%161) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %163 = "mhlo.get_tuple_element"(%161) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %164 = "mhlo.get_tuple_element"(%41) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %165 = "mhlo.get_tuple_element"(%41) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %166 = "mhlo.custom_call"(%163, %40, %arg15, %164, %165) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %167 = "mhlo.get_tuple_element"(%166) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %168 = "mhlo.reshape"(%167) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %169 = "mhlo.custom_call"(%168) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %170 = "mhlo.custom_call"(%169, %36, %37) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %171 = "mhlo.get_tuple_element"(%170) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %172 = "mhlo.get_tuple_element"(%35) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %173 = "mhlo.get_tuple_element"(%35) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %174 = "mhlo.custom_call"(%171, %172, %173) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %175 = "mhlo.custom_call"(%174, %32, %33) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %176 = "mhlo.get_tuple_element"(%175) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %177 = "mhlo.custom_call"(%176, %31, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %178 = "mhlo.get_tuple_element"(%177) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %179 = mhlo.add %162, %178 : tensor<2x128x128xf32>
    %180 = "mhlo.get_tuple_element"(%170) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %181 = "mhlo.custom_call"(%180, %31, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %182 = "mhlo.get_tuple_element"(%181) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %183 = mhlo.add %179, %182 : tensor<2x128x128xf32>
    %184 = "mhlo.get_tuple_element"(%175) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %185 = "mhlo.custom_call"(%184, %31, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %186 = "mhlo.get_tuple_element"(%185) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %187 = mhlo.add %183, %186 : tensor<2x128x128xf32>
    %188 = "mhlo.get_tuple_element"(%30) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %189 = "mhlo.get_tuple_element"(%30) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %190 = "mhlo.get_tuple_element"(%30) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %191 = "mhlo.custom_call"(%187, %188, %arg6, %189, %190) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %192 = "mhlo.get_tuple_element"(%191) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %193 = "mhlo.reshape"(%192) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %194 = "mhlo.select"(%87, %193, %6) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %195 = "mhlo.scatter"(%80, %84, %194) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %256 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%256) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %196 = "mhlo.compare"(%20, %9) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %197 = mhlo.add %20, %8 : tensor<256xi64>
    %198 = "mhlo.select"(%196, %197, %20) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %199 = "mhlo.reshape"(%198) : (tensor<256xi64>) -> tensor<256x1xi64>
    %200 = "mhlo.convert"(%20) : (tensor<256xi64>) -> tensor<256xf64>
    %201 = "mhlo.compare"(%200, %7) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    %202 = "mhlo.broadcast_in_dim"(%201) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %203 = "mhlo.select"(%202, %193, %6) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %204 = "mhlo.scatter"(%10, %199, %203) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %256 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%256) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %205 = "mhlo.compare"(%26, %4) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %206 = mhlo.add %26, %3 : tensor<128xi64>
    %207 = "mhlo.select"(%205, %206, %26) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %208 = "mhlo.reshape"(%207) : (tensor<128xi64>) -> tensor<128x1xi64>
    %209 = "mhlo.convert"(%26) : (tensor<128xi64>) -> tensor<128xf64>
    %210 = "mhlo.compare"(%209, %2) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    %211 = "mhlo.broadcast_in_dim"(%210) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %212 = "mhlo.get_tuple_element"(%191) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %213 = "mhlo.reduce"(%212, %0) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %256 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%256) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %214 = "mhlo.select"(%211, %213, %1) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %215 = "mhlo.scatter"(%5, %208, %214) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %256 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%256) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %216 = "mhlo.get_tuple_element"(%191) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %217 = "mhlo.get_tuple_element"(%191) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %218 = "mhlo.get_tuple_element"(%177) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %219 = "mhlo.get_tuple_element"(%177) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %220 = "mhlo.get_tuple_element"(%185) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %221 = "mhlo.get_tuple_element"(%185) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %222 = "mhlo.get_tuple_element"(%181) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %223 = "mhlo.get_tuple_element"(%181) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %224 = "mhlo.get_tuple_element"(%166) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %225 = "mhlo.get_tuple_element"(%166) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %226 = "mhlo.get_tuple_element"(%161) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %227 = "mhlo.get_tuple_element"(%161) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %228 = "mhlo.get_tuple_element"(%155) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %229 = "mhlo.get_tuple_element"(%155) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %230 = "mhlo.get_tuple_element"(%151) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %231 = "mhlo.get_tuple_element"(%151) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %232 = "mhlo.get_tuple_element"(%146) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %233 = "mhlo.get_tuple_element"(%146) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %234 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %235 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %236 = "mhlo.get_tuple_element"(%140) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %237 = "mhlo.get_tuple_element"(%140) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %238 = "mhlo.get_tuple_element"(%136) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %239 = "mhlo.get_tuple_element"(%136) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %240 = "mhlo.get_tuple_element"(%121) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %241 = "mhlo.get_tuple_element"(%121) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %242 = "mhlo.get_tuple_element"(%116) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %243 = "mhlo.get_tuple_element"(%116) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %244 = "mhlo.get_tuple_element"(%110) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %245 = "mhlo.get_tuple_element"(%110) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %246 = "mhlo.get_tuple_element"(%106) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %247 = "mhlo.get_tuple_element"(%106) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %248 = "mhlo.get_tuple_element"(%101) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %249 = "mhlo.get_tuple_element"(%101) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %250 = "mhlo.get_tuple_element"(%96) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %251 = "mhlo.get_tuple_element"(%96) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %252 = "mhlo.get_tuple_element"(%92) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %253 = "mhlo.get_tuple_element"(%92) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %254 = "mhlo.reduce"(%arg47, %0) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %256 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%256) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %255 = "mhlo.tuple"(%78, %195, %204, %215, %216, %217, %218, %219, %220, %221, %222, %223, %224, %225, %226, %227, %228, %229, %230, %231, %232, %233, %234, %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250, %251, %252, %253, %254) : (tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) -> !tuple
    return %255 : !tuple
  }
}

