// RUN: byteir-opt %s -inline -mhlo-arith-opt -cse -sccp -canonicalize | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>>
module  {
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<1x512xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<30522x128xf32>, %arg4: tensor<2x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<2x1x1x128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<2x1x1x128xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<512x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<30522xf32>, %arg47: tensor<2x128x30522xf32>) -> !tuple {
    %0 = call @aten.view.81(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %1 = call @aten.index_select.101(%arg3, %0) : (tensor<30522x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %2 = call @aten.view.91(%1) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %3 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %4 = "mhlo.slice"(%3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %5 = call @aten.expand.75(%4) : (tensor<1x128xi64>) -> tensor<2x128xi64>
    %6 = call @aten.view.81(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %7 = call @aten.index_select.85(%arg4, %6) : (tensor<2x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %8 = call @aten.view.91(%7) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %9 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = call @aten.expand.66(%9) : (tensor<f32>) -> tensor<2x128x128xf32>
    %11 = call @aten.mul.95(%8, %10) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %12 = call @aten.add.108(%2, %11) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %13 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %14 = "mhlo.slice"(%13) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %15 = call @aten.view.51(%14) : (tensor<1x128xi64>) -> tensor<128xi64>
    %16 = call @aten.index_select.55(%arg5, %15) : (tensor<512x128xf32>, tensor<128xi64>) -> tensor<128x128xf32>
    %17 = call @aten.view.61(%16) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %18 = "mhlo.custom_call"(%12, %arg6, %arg7, %17) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %19 = "mhlo.get_tuple_element"(%18) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %20 = "mhlo.custom_call"(%19, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %21 = "mhlo.custom_call"(%19, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %22 = "mhlo.custom_call"(%20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %23 = "mhlo.custom_call"(%22, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %24 = "mhlo.get_tuple_element"(%23) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %25 = "mhlo.custom_call"(%19, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %26 = "mhlo.custom_call"(%24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %27 = "mhlo.custom_call"(%26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %28 = call @aten.view.128(%27) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %29 = "mhlo.custom_call"(%28, %arg15, %arg16) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %30 = "mhlo.get_tuple_element"(%29) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %31 = "mhlo.custom_call"(%30, %arg17, %arg18, %19) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %32 = "mhlo.get_tuple_element"(%31) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %33 = "mhlo.custom_call"(%32, %arg19, %arg20) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %35 = "mhlo.custom_call"(%34, %arg21, %arg22) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %36 = "mhlo.get_tuple_element"(%35) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %37 = "mhlo.custom_call"(%36, %arg23, %arg24, %32) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %38 = "mhlo.get_tuple_element"(%37) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %39 = "mhlo.custom_call"(%38, %arg25, %arg26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %40 = "mhlo.custom_call"(%38, %arg27, %arg28) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %41 = "mhlo.custom_call"(%39, %40) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %42 = "mhlo.custom_call"(%41, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %44 = "mhlo.custom_call"(%38, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %45 = "mhlo.custom_call"(%43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %46 = "mhlo.custom_call"(%45) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %47 = call @aten.view.128(%46) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %48 = "mhlo.custom_call"(%47, %arg32, %arg33) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %50 = "mhlo.custom_call"(%49, %arg34, %arg35, %38) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %51 = "mhlo.get_tuple_element"(%50) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %52 = "mhlo.custom_call"(%51, %arg36, %arg37) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>
    %53 = "mhlo.get_tuple_element"(%52) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %54 = "mhlo.custom_call"(%53, %arg38, %arg39) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>
    %55 = "mhlo.get_tuple_element"(%54) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %56 = "mhlo.custom_call"(%55, %arg40, %arg41, %51) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %57 = "mhlo.get_tuple_element"(%56) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %58 = "mhlo.custom_call"(%57, %arg42, %arg43) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>
    %59 = "mhlo.get_tuple_element"(%58) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %60 = "mhlo.custom_call"(%59, %arg44, %arg45) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>
    %61 = "mhlo.get_tuple_element"(%60) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<2x128x128xf32>
    %62 = "mhlo.custom_call"(%61, %arg3, %arg46) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %63 = "mhlo.custom_call"(%arg47, %61, %arg3) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false} : (tensor<2x128x30522xf32>, tensor<2x128x128xf32>, tensor<30522x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>
    %64 = "mhlo.get_tuple_element"(%63) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<30522x128xf32>
    %65 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = call @aten.expand.197(%65) : (tensor<f32>) -> tensor<30522x128xf32>
    %67 = call @aten.view.81(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %68 = mhlo.constant dense<0> : tensor<i64>
    %69 = call @aten.lt.393(%67, %68) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %70 = mhlo.constant dense<30522> : tensor<i64>
    %71 = call @aten.expand.380(%70) : (tensor<i64>) -> tensor<256xi64>
    %72 = call @aten.add.387(%67, %71) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %73 = call @aten.where.399(%69, %72, %67) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %74 = call @aten.stack.405(%73) : (tensor<256xi64>) -> tensor<256x1xi64>
    %75 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %76 = call @aten.ne.356(%67, %75) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %77 = call @aten.view.363(%76) : (tensor<256xi1>) -> tensor<256x1xi1>
    %78 = call @aten.expand.367(%77) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %79 = "mhlo.get_tuple_element"(%63) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<2x128x128xf32>
    %80 = "mhlo.get_tuple_element"(%60) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %81 = "mhlo.get_tuple_element"(%60) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %82 = "mhlo.custom_call"(%79, %59, %arg44, %80, %81) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %83 = "mhlo.get_tuple_element"(%82) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %84 = "mhlo.get_tuple_element"(%58) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %85 = "mhlo.get_tuple_element"(%58) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %86 = "mhlo.custom_call"(%83, %57, %arg42, %84, %85) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %87 = "mhlo.get_tuple_element"(%86) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %88 = "mhlo.get_tuple_element"(%56) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %89 = "mhlo.get_tuple_element"(%56) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %90 = "mhlo.get_tuple_element"(%56) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %91 = "mhlo.custom_call"(%87, %88, %arg40, %89, %90) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %92 = "mhlo.get_tuple_element"(%91) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %93 = "mhlo.get_tuple_element"(%91) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %94 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %95 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %96 = "mhlo.custom_call"(%93, %53, %arg38, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %97 = "mhlo.get_tuple_element"(%96) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %98 = "mhlo.get_tuple_element"(%52) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %99 = "mhlo.get_tuple_element"(%52) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %100 = "mhlo.custom_call"(%97, %51, %arg36, %98, %99) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %101 = "mhlo.get_tuple_element"(%100) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %102 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %103 = call @aten.expand.66(%102) : (tensor<f32>) -> tensor<2x128x128xf32>
    %104 = call @aten.mul.95(%101, %103) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %105 = call @aten.add.108(%92, %104) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %106 = "mhlo.get_tuple_element"(%50) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %107 = "mhlo.get_tuple_element"(%50) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %108 = "mhlo.get_tuple_element"(%50) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %109 = "mhlo.custom_call"(%105, %106, %arg34, %107, %108) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %110 = "mhlo.get_tuple_element"(%109) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %111 = "mhlo.get_tuple_element"(%109) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %112 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %113 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %114 = "mhlo.custom_call"(%111, %47, %arg32, %112, %113) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %115 = "mhlo.get_tuple_element"(%114) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %116 = call @aten.view.256(%115) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %117 = "mhlo.custom_call"(%116) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %118 = "mhlo.custom_call"(%117, %43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %119 = "mhlo.get_tuple_element"(%118) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %120 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %121 = "mhlo.get_tuple_element"(%42) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %122 = "mhlo.custom_call"(%119, %120, %121) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %123 = "mhlo.custom_call"(%122, %39, %40) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %124 = "mhlo.get_tuple_element"(%123) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %125 = "mhlo.custom_call"(%124, %38, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %126 = "mhlo.get_tuple_element"(%125) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %127 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %128 = call @aten.expand.66(%127) : (tensor<f32>) -> tensor<2x128x128xf32>
    %129 = call @aten.mul.95(%126, %128) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %130 = call @aten.add.108(%110, %129) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %131 = "mhlo.get_tuple_element"(%118) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %132 = "mhlo.custom_call"(%131, %38, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %133 = "mhlo.get_tuple_element"(%132) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %134 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %135 = call @aten.expand.66(%134) : (tensor<f32>) -> tensor<2x128x128xf32>
    %136 = call @aten.mul.95(%133, %135) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %137 = call @aten.add.108(%130, %136) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %138 = "mhlo.get_tuple_element"(%123) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %139 = "mhlo.custom_call"(%138, %38, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %140 = "mhlo.get_tuple_element"(%139) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %141 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %142 = call @aten.expand.66(%141) : (tensor<f32>) -> tensor<2x128x128xf32>
    %143 = call @aten.mul.95(%140, %142) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %144 = call @aten.add.108(%137, %143) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %145 = "mhlo.get_tuple_element"(%37) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %146 = "mhlo.get_tuple_element"(%37) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %147 = "mhlo.get_tuple_element"(%37) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %148 = "mhlo.custom_call"(%144, %145, %arg23, %146, %147) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %149 = "mhlo.get_tuple_element"(%148) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %150 = "mhlo.get_tuple_element"(%148) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %151 = "mhlo.get_tuple_element"(%35) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %152 = "mhlo.get_tuple_element"(%35) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %153 = "mhlo.custom_call"(%150, %34, %arg21, %151, %152) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %154 = "mhlo.get_tuple_element"(%153) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %155 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %156 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %157 = "mhlo.custom_call"(%154, %32, %arg19, %155, %156) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %158 = "mhlo.get_tuple_element"(%157) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %159 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = call @aten.expand.66(%159) : (tensor<f32>) -> tensor<2x128x128xf32>
    %161 = call @aten.mul.95(%158, %160) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %162 = call @aten.add.108(%149, %161) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %163 = "mhlo.get_tuple_element"(%31) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %164 = "mhlo.get_tuple_element"(%31) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %165 = "mhlo.get_tuple_element"(%31) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %166 = "mhlo.custom_call"(%162, %163, %arg17, %164, %165) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %167 = "mhlo.get_tuple_element"(%166) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %168 = "mhlo.get_tuple_element"(%166) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %169 = "mhlo.get_tuple_element"(%29) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %170 = "mhlo.get_tuple_element"(%29) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %171 = "mhlo.custom_call"(%168, %28, %arg15, %169, %170) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %172 = "mhlo.get_tuple_element"(%171) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %173 = call @aten.view.256(%172) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %174 = "mhlo.custom_call"(%173) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %175 = "mhlo.custom_call"(%174, %24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %176 = "mhlo.get_tuple_element"(%175) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %177 = "mhlo.get_tuple_element"(%23) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %178 = "mhlo.get_tuple_element"(%23) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %179 = "mhlo.custom_call"(%176, %177, %178) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %180 = "mhlo.custom_call"(%179, %20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %181 = "mhlo.get_tuple_element"(%180) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %182 = "mhlo.custom_call"(%181, %19, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %183 = "mhlo.get_tuple_element"(%182) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %184 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %185 = call @aten.expand.66(%184) : (tensor<f32>) -> tensor<2x128x128xf32>
    %186 = call @aten.mul.95(%183, %185) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %187 = call @aten.add.108(%167, %186) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %188 = "mhlo.get_tuple_element"(%175) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %189 = "mhlo.custom_call"(%188, %19, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %190 = "mhlo.get_tuple_element"(%189) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %191 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %192 = call @aten.expand.66(%191) : (tensor<f32>) -> tensor<2x128x128xf32>
    %193 = call @aten.mul.95(%190, %192) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %194 = call @aten.add.108(%187, %193) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %195 = "mhlo.get_tuple_element"(%180) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %196 = "mhlo.custom_call"(%195, %19, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %197 = "mhlo.get_tuple_element"(%196) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %198 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %199 = call @aten.expand.66(%198) : (tensor<f32>) -> tensor<2x128x128xf32>
    %200 = call @aten.mul.95(%197, %199) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %201 = call @aten.add.108(%194, %200) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %202 = "mhlo.get_tuple_element"(%18) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %203 = "mhlo.get_tuple_element"(%18) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %204 = "mhlo.get_tuple_element"(%18) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %205 = "mhlo.custom_call"(%201, %202, %arg6, %203, %204) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %206 = "mhlo.get_tuple_element"(%205) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %207 = call @aten.view.350(%206) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %208 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = call @aten.expand.205(%208) : (tensor<f32>) -> tensor<256x128xf32>
    %210 = call @aten.where.373(%78, %207, %209) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %211 = call @aten.index_put.416(%66, %74, %210) : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %212 = call @aten.permute.423(%211) : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %213 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %214 = call @aten.expand.197(%213) : (tensor<f32>) -> tensor<30522x128xf32>
    %215 = call @aten.mul.427(%212, %214) : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %216 = call @aten.add.432(%64, %215) : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %217 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %218 = call @aten.expand.454(%217) : (tensor<f32>) -> tensor<2x128xf32>
    %219 = call @aten.view.81(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %220 = mhlo.constant dense<0> : tensor<i64>
    %221 = call @aten.lt.393(%219, %220) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %222 = mhlo.constant dense<2> : tensor<i64>
    %223 = call @aten.expand.380(%222) : (tensor<i64>) -> tensor<256xi64>
    %224 = call @aten.add.387(%219, %223) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %225 = call @aten.where.399(%221, %224, %219) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %226 = call @aten.stack.405(%225) : (tensor<256xi64>) -> tensor<256x1xi64>
    %227 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %228 = call @aten.ne.356(%219, %227) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %229 = call @aten.view.363(%228) : (tensor<256xi1>) -> tensor<256x1xi1>
    %230 = call @aten.expand.367(%229) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %231 = call @aten.view.350(%206) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %232 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %233 = call @aten.expand.205(%232) : (tensor<f32>) -> tensor<256x128xf32>
    %234 = call @aten.where.373(%230, %231, %233) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %235 = call @aten.index_put.465(%218, %226, %234) : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %236 = call @aten.permute.472(%235) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %237 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %238 = call @aten.expand.558(%237) : (tensor<f32>) -> tensor<512x128xf32>
    %239 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %240 = "mhlo.slice"(%239) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %241 = call @aten.view.51(%240) : (tensor<1x128xi64>) -> tensor<128xi64>
    %242 = mhlo.constant dense<0> : tensor<i64>
    %243 = call @aten.lt.540(%241, %242) : (tensor<128xi64>, tensor<i64>) -> tensor<128xi1>
    %244 = mhlo.constant dense<512> : tensor<i64>
    %245 = call @aten.expand.527(%244) : (tensor<i64>) -> tensor<128xi64>
    %246 = call @aten.add.534(%241, %245) : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %247 = call @aten.where.546(%243, %246, %241) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %248 = call @aten.stack.552(%247) : (tensor<128xi64>) -> tensor<128x1xi64>
    %249 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %250 = call @aten.ne.503(%241, %249) : (tensor<128xi64>, tensor<f64>) -> tensor<128xi1>
    %251 = call @aten.view.510(%250) : (tensor<128xi1>) -> tensor<128x1xi1>
    %252 = call @aten.expand.514(%251) : (tensor<128x1xi1>) -> tensor<128x128xi1>
    %253 = "mhlo.get_tuple_element"(%205) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %254 = call @aten.sum.488(%253) : (tensor<2x128x128xf32>) -> tensor<1x128x128xf32>
    %255 = call @aten.view.495(%254) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %256 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %257 = call @aten.expand.477(%256) : (tensor<f32>) -> tensor<128x128xf32>
    %258 = call @aten.where.520(%252, %255, %257) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %259 = call @aten.index_put.569(%238, %248, %258) : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %260 = call @aten.permute.576(%259) : (tensor<512x128xf32>) -> tensor<512x128xf32>
    %261 = "mhlo.get_tuple_element"(%205) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %262 = "mhlo.get_tuple_element"(%205) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %263 = "mhlo.get_tuple_element"(%182) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %264 = "mhlo.get_tuple_element"(%182) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %265 = "mhlo.get_tuple_element"(%196) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %266 = "mhlo.get_tuple_element"(%196) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %267 = "mhlo.get_tuple_element"(%189) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %268 = "mhlo.get_tuple_element"(%189) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %269 = "mhlo.get_tuple_element"(%171) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %270 = "mhlo.get_tuple_element"(%171) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %271 = "mhlo.get_tuple_element"(%166) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %272 = "mhlo.get_tuple_element"(%166) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %273 = "mhlo.get_tuple_element"(%157) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %274 = "mhlo.get_tuple_element"(%157) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %275 = "mhlo.get_tuple_element"(%153) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %276 = "mhlo.get_tuple_element"(%153) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %277 = "mhlo.get_tuple_element"(%148) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %278 = "mhlo.get_tuple_element"(%148) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %279 = "mhlo.get_tuple_element"(%125) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %280 = "mhlo.get_tuple_element"(%125) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %281 = "mhlo.get_tuple_element"(%139) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %282 = "mhlo.get_tuple_element"(%139) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %283 = "mhlo.get_tuple_element"(%132) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %284 = "mhlo.get_tuple_element"(%132) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %285 = "mhlo.get_tuple_element"(%114) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %286 = "mhlo.get_tuple_element"(%114) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %287 = "mhlo.get_tuple_element"(%109) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %288 = "mhlo.get_tuple_element"(%109) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %289 = "mhlo.get_tuple_element"(%100) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %290 = "mhlo.get_tuple_element"(%100) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %291 = "mhlo.get_tuple_element"(%96) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %292 = "mhlo.get_tuple_element"(%96) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %293 = "mhlo.get_tuple_element"(%91) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %294 = "mhlo.get_tuple_element"(%91) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %295 = "mhlo.get_tuple_element"(%86) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %296 = "mhlo.get_tuple_element"(%86) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %297 = "mhlo.get_tuple_element"(%82) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %298 = "mhlo.get_tuple_element"(%82) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %299 = "mhlo.get_tuple_element"(%63) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<30522x128xf32>, tensor<30522xf32>>) -> tensor<30522xf32>
    %300 = "mhlo.tuple"(%62, %216, %236, %260, %261, %262, %263, %264, %265, %266, %267, %268, %269, %270, %271, %272, %273, %274, %275, %276, %277, %278, %279, %280, %281, %282, %283, %284, %285, %286, %287, %288, %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299) : (tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) -> !tuple
    return %300 : !tuple
  }
  func private @aten.view.81(%arg0: tensor<2x128xi64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.index_select.101(%arg0: tensor<30522x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func private @aten.view.91(%arg0: tensor<256x128xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.expand.75(%arg0: tensor<1x128xi64>) -> tensor<2x128xi64> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xi64>) -> tensor<1x128xi64>
    %1 = "mhlo.reshape"(%0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    return %2 : tensor<2x128xi64>
  }
  func private @aten.index_select.85(%arg0: tensor<2x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func private @aten.expand.66(%arg0: tensor<f32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func private @aten.mul.95(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.add.108(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.view.51(%arg0: tensor<1x128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.index_select.55(%arg0: tensor<512x128xf32>, %arg1: tensor<128xi64>) -> tensor<128x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<128xi64>) -> tensor<128xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @aten.view.61(%arg0: tensor<128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
  func private @aten.view.128(%arg0: tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.expand.197(%arg0: tensor<f32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<30522x128xf32>
    return %3 : tensor<30522x128xf32>
  }
  func private @aten.lt.393(%arg0: tensor<256xi64>, %arg1: tensor<i64>) -> tensor<256xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    return %1 : tensor<256xi1>
  }
  func private @aten.expand.380(%arg0: tensor<i64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    return %3 : tensor<256xi64>
  }
  func private @aten.add.387(%arg0: tensor<256xi64>, %arg1: tensor<256xi64>) -> tensor<256xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.where.399(%arg0: tensor<256xi1>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>) -> tensor<256xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.stack.405(%arg0: tensor<256xi64>) -> tensor<256x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi64>) -> tensor<256x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<256x1xi64>) -> tensor<256x1xi64>
    return %1 : tensor<256x1xi64>
  }
  func private @aten.ne.356(%arg0: tensor<256xi64>, %arg1: tensor<f64>) -> tensor<256xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<256xi64>) -> tensor<256xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<256xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %2 : tensor<256xi1>
  }
  func private @aten.view.363(%arg0: tensor<256xi1>) -> tensor<256x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi1>) -> tensor<256x1xi1>
    return %0 : tensor<256x1xi1>
  }
  func private @aten.expand.367(%arg0: tensor<256x1xi1>) -> tensor<256x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x1xi1>) -> tensor<256x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<256x1xi1>) -> tensor<256xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    return %2 : tensor<256x128xi1>
  }
  func private @aten.view.256(%arg0: tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    return %0 : tensor<2x128x2x64xf32>
  }
  func private @aten.view.350(%arg0: tensor<2x128x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func private @aten.expand.205(%arg0: tensor<f32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x128xf32>
    return %3 : tensor<256x128xf32>
  }
  func private @aten.where.373(%arg0: tensor<256x128xi1>, %arg1: tensor<256x128xf32>, %arg2: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func private @aten.index_put.416(%arg0: tensor<30522x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @aten.permute.423(%arg0: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.mul.427(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.add.432(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.expand.454(%arg0: tensor<f32>) -> tensor<2x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    return %3 : tensor<2x128xf32>
  }
  func private @aten.index_put.465(%arg0: tensor<2x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    return %1 : tensor<2x128xf32>
  }
  func private @aten.permute.472(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128xf32>) -> tensor<2x128xf32>
    return %0 : tensor<2x128xf32>
  }
  func private @aten.expand.558(%arg0: tensor<f32>) -> tensor<512x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512x128xf32>
    return %3 : tensor<512x128xf32>
  }
  func private @aten.lt.540(%arg0: tensor<128xi64>, %arg1: tensor<i64>) -> tensor<128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    return %1 : tensor<128xi1>
  }
  func private @aten.expand.527(%arg0: tensor<i64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    return %3 : tensor<128xi64>
  }
  func private @aten.add.534(%arg0: tensor<128xi64>, %arg1: tensor<128xi64>) -> tensor<128xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.where.546(%arg0: tensor<128xi1>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.stack.552(%arg0: tensor<128xi64>) -> tensor<128x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi64>) -> tensor<128x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<128x1xi64>) -> tensor<128x1xi64>
    return %1 : tensor<128x1xi64>
  }
  func private @aten.ne.503(%arg0: tensor<128xi64>, %arg1: tensor<f64>) -> tensor<128xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<128xi64>) -> tensor<128xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<128xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %2 : tensor<128xi1>
  }
  func private @aten.view.510(%arg0: tensor<128xi1>) -> tensor<128x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi1>) -> tensor<128x1xi1>
    return %0 : tensor<128x1xi1>
  }
  func private @aten.expand.514(%arg0: tensor<128x1xi1>) -> tensor<128x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x1xi1>) -> tensor<128x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<128x1xi1>) -> tensor<128xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    return %2 : tensor<128x128xi1>
  }
  func private @aten.sum.488(%arg0: tensor<2x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = mhlo.constant dense<2> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.reduce"(%arg0, %1) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    return %3 : tensor<1x128x128xf32>
  }
  func private @aten.view.495(%arg0: tensor<1x128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func private @aten.expand.477(%arg0: tensor<f32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128x128xf32>
    return %3 : tensor<128x128xf32>
  }
  func private @aten.where.520(%arg0: tensor<128x128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func private @aten.index_put.569(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }
  func private @aten.permute.576(%arg0: tensor<512x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<512x128xf32>
    return %0 : tensor<512x128xf32>
  }
}
