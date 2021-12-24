// RUN: byteir-opt %s -inline -mhlo-arith-opt -cse -sccp -canonicalize | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>>
module  {
  func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<1x512xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<30522x128xf32>, %arg4: tensor<2x128xf32>, %arg5: tensor<512x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<2x1x1x128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<2x1x1x128xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<512x128xf32>, %arg37: tensor<512xf32>, %arg38: tensor<128x512xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<30522xf32>, %arg47: tensor<2x128x30522xf32>) -> !tuple {
    %0 = call @aten.view.98(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %1 = call @aten.index_select.118(%arg3, %0) : (tensor<30522x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %2 = call @aten.view.108(%1) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %3 = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %4 = "mhlo.slice"(%3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %5 = call @aten.expand.92(%4) : (tensor<1x128xi64>) -> tensor<2x128xi64>
    %6 = call @aten.view.98(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %7 = call @aten.index_select.102(%arg4, %6) : (tensor<2x128xf32>, tensor<256xi64>) -> tensor<256x128xf32>
    %8 = call @aten.view.108(%7) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %9 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %10 = call @aten.expand.83(%9) : (tensor<f32>) -> tensor<2x128x128xf32>
    %11 = call @aten.mul.112(%8, %10) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %12 = call @aten.add.125(%2, %11) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %13 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %14 = "mhlo.slice"(%13) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %15 = call @aten.view.68(%14) : (tensor<1x128xi64>) -> tensor<128xi64>
    %16 = call @aten.index_select.72(%arg5, %15) : (tensor<512x128xf32>, tensor<128xi64>) -> tensor<128x128xf32>
    %17 = call @aten.view.78(%16) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %18 = "mhlo.custom_call"(%12, %arg6, %arg7, %17) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1x128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>
    %19 = "mhlo.get_tuple_element"(%18) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %20 = "mhlo.custom_call"(%19, %arg8, %arg9) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %21 = "mhlo.custom_call"(%19, %arg10, %arg11) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %22 = "mhlo.custom_call"(%20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %23 = "mhlo.custom_call"(%22, %arg14) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %24 = "mhlo.get_tuple_element"(%23) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %25 = "mhlo.custom_call"(%19, %arg12, %arg13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %26 = "mhlo.custom_call"(%24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %27 = "mhlo.custom_call"(%26) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %28 = call @aten.view.145(%27) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
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
    %42 = "mhlo.custom_call"(%41, %arg31) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>
    %43 = "mhlo.get_tuple_element"(%42) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %44 = "mhlo.custom_call"(%38, %arg29, %arg30) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %45 = "mhlo.custom_call"(%43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %46 = "mhlo.custom_call"(%45) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %47 = call @aten.view.145(%46) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
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
    %62 = call @aten.view.212(%61) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %63 = call @aten.permute.62(%arg3) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %64 = call @aten.mm.216(%62, %63) : (tensor<256x128xf32>, tensor<128x30522xf32>) -> tensor<256x30522xf32>
    %65 = call @aten.view.221(%64) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %66 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %67 = call @aten.expand.50(%66) : (tensor<f32>) -> tensor<30522xf32>
    %68 = call @aten.mul.57(%arg46, %67) : (tensor<30522xf32>, tensor<30522xf32>) -> tensor<30522xf32>
    %69 = call @aten.add.225(%65, %68) : (tensor<2x128x30522xf32>, tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %70 = call @aten.view.231(%69) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %71 = call @aten.view.221(%70) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %72 = call @aten.view.212(%61) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %73 = call @aten.permute.478(%72) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %74 = call @aten.view.231(%arg47) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    %75 = call @aten.mm.482(%73, %74) : (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %76 = call @aten.permute.487(%75) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %77 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %78 = call @aten.expand.237(%77) : (tensor<f32>) -> tensor<30522x128xf32>
    %79 = call @aten.view.98(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    %80 = mhlo.constant dense<0> : tensor<i64>
    %81 = call @aten.lt.438(%79, %80) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %82 = mhlo.constant dense<30522> : tensor<i64>
    %83 = call @aten.expand.425(%82) : (tensor<i64>) -> tensor<256xi64>
    %84 = call @aten.add.432(%79, %83) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %85 = call @aten.where.444(%81, %84, %79) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %86 = call @aten.stack.450(%85) : (tensor<256xi64>) -> tensor<256x1xi64>
    %87 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %88 = call @aten.ne.401(%79, %87) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %89 = call @aten.view.408(%88) : (tensor<256xi1>) -> tensor<256x1xi1>
    %90 = call @aten.expand.412(%89) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %91 = call @aten.permute.62(%arg3) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %92 = call @aten.permute.261(%91) : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    %93 = call @aten.mm.266(%74, %92) : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %94 = call @aten.view.108(%93) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %95 = "mhlo.get_tuple_element"(%60) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %96 = "mhlo.get_tuple_element"(%60) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %97 = "mhlo.custom_call"(%94, %59, %arg44, %95, %96) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>
    %98 = "mhlo.get_tuple_element"(%97) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %99 = "mhlo.get_tuple_element"(%58) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<2x128x128xf32>
    %100 = "mhlo.get_tuple_element"(%58) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %101 = "mhlo.custom_call"(%98, %57, %arg42, %99, %100) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %102 = "mhlo.get_tuple_element"(%101) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %103 = "mhlo.get_tuple_element"(%56) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %104 = "mhlo.get_tuple_element"(%56) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %105 = "mhlo.get_tuple_element"(%56) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %106 = "mhlo.custom_call"(%102, %103, %arg40, %104, %105) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %107 = "mhlo.get_tuple_element"(%106) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %108 = "mhlo.get_tuple_element"(%106) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %109 = "mhlo.get_tuple_element"(%54) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %110 = "mhlo.get_tuple_element"(%54) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %111 = "mhlo.custom_call"(%108, %53, %arg38, %109, %110) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %112 = "mhlo.get_tuple_element"(%111) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %113 = "mhlo.get_tuple_element"(%52) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %114 = "mhlo.get_tuple_element"(%52) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %115 = "mhlo.custom_call"(%112, %51, %arg36, %113, %114) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %116 = "mhlo.get_tuple_element"(%115) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %117 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %118 = call @aten.expand.83(%117) : (tensor<f32>) -> tensor<2x128x128xf32>
    %119 = call @aten.mul.112(%116, %118) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %120 = call @aten.add.125(%107, %119) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %121 = "mhlo.get_tuple_element"(%50) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %122 = "mhlo.get_tuple_element"(%50) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %123 = "mhlo.get_tuple_element"(%50) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %124 = "mhlo.custom_call"(%120, %121, %arg34, %122, %123) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %125 = "mhlo.get_tuple_element"(%124) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %126 = "mhlo.get_tuple_element"(%124) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %127 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %128 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %129 = "mhlo.custom_call"(%126, %47, %arg32, %127, %128) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %130 = "mhlo.get_tuple_element"(%129) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %131 = call @aten.view.304(%130) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %132 = "mhlo.custom_call"(%131) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %133 = "mhlo.custom_call"(%132, %43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %134 = "mhlo.get_tuple_element"(%133) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %135 = "mhlo.get_tuple_element"(%42) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %136 = "mhlo.get_tuple_element"(%42) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %137 = "mhlo.custom_call"(%134, %135, %136) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %138 = "mhlo.custom_call"(%137, %39, %40) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %139 = "mhlo.get_tuple_element"(%138) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %140 = "mhlo.custom_call"(%139, %38, %arg25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %141 = "mhlo.get_tuple_element"(%140) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %142 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %143 = call @aten.expand.83(%142) : (tensor<f32>) -> tensor<2x128x128xf32>
    %144 = call @aten.mul.112(%141, %143) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %145 = call @aten.add.125(%125, %144) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %146 = "mhlo.get_tuple_element"(%133) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %147 = "mhlo.custom_call"(%146, %38, %arg29) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %148 = "mhlo.get_tuple_element"(%147) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %149 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %150 = call @aten.expand.83(%149) : (tensor<f32>) -> tensor<2x128x128xf32>
    %151 = call @aten.mul.112(%148, %150) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %152 = call @aten.add.125(%145, %151) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %153 = "mhlo.get_tuple_element"(%138) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %154 = "mhlo.custom_call"(%153, %38, %arg27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %155 = "mhlo.get_tuple_element"(%154) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %156 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %157 = call @aten.expand.83(%156) : (tensor<f32>) -> tensor<2x128x128xf32>
    %158 = call @aten.mul.112(%155, %157) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %159 = call @aten.add.125(%152, %158) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %160 = "mhlo.get_tuple_element"(%37) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %161 = "mhlo.get_tuple_element"(%37) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %162 = "mhlo.get_tuple_element"(%37) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %163 = "mhlo.custom_call"(%159, %160, %arg23, %161, %162) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %164 = "mhlo.get_tuple_element"(%163) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %165 = "mhlo.get_tuple_element"(%163) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %166 = "mhlo.get_tuple_element"(%35) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %167 = "mhlo.get_tuple_element"(%35) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %168 = "mhlo.custom_call"(%165, %34, %arg21, %166, %167) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>
    %169 = "mhlo.get_tuple_element"(%168) {index = 0 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<2x128x512xf32>
    %170 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<2x128x512xf32>
    %171 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>>) -> tensor<0xf32>
    %172 = "mhlo.custom_call"(%169, %32, %arg19, %170, %171) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>
    %173 = "mhlo.get_tuple_element"(%172) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<2x128x128xf32>
    %174 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %175 = call @aten.expand.83(%174) : (tensor<f32>) -> tensor<2x128x128xf32>
    %176 = call @aten.mul.112(%173, %175) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %177 = call @aten.add.125(%164, %176) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %178 = "mhlo.get_tuple_element"(%31) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %179 = "mhlo.get_tuple_element"(%31) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %180 = "mhlo.get_tuple_element"(%31) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %181 = "mhlo.custom_call"(%177, %178, %arg17, %179, %180) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %182 = "mhlo.get_tuple_element"(%181) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %183 = "mhlo.get_tuple_element"(%181) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %184 = "mhlo.get_tuple_element"(%29) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xf32>
    %185 = "mhlo.get_tuple_element"(%29) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>>) -> tensor<2x128x128xui8>
    %186 = "mhlo.custom_call"(%183, %28, %arg15, %184, %185) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xui8>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %187 = "mhlo.get_tuple_element"(%186) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %188 = call @aten.view.304(%187) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %189 = "mhlo.custom_call"(%188) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %190 = "mhlo.custom_call"(%189, %24, %25) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>
    %191 = "mhlo.get_tuple_element"(%190) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x128xf32>
    %192 = "mhlo.get_tuple_element"(%23) {index = 0 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xf32>
    %193 = "mhlo.get_tuple_element"(%23) {index = 2 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>>) -> tensor<2x2x128x128xui8>
    %194 = "mhlo.custom_call"(%191, %192, %193) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %195 = "mhlo.custom_call"(%194, %20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>
    %196 = "mhlo.get_tuple_element"(%195) {index = 0 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %197 = "mhlo.custom_call"(%196, %19, %arg8) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %198 = "mhlo.get_tuple_element"(%197) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %199 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %200 = call @aten.expand.83(%199) : (tensor<f32>) -> tensor<2x128x128xf32>
    %201 = call @aten.mul.112(%198, %200) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %202 = call @aten.add.125(%182, %201) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %203 = "mhlo.get_tuple_element"(%190) {index = 1 : i32} : (tuple<tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %204 = "mhlo.custom_call"(%203, %19, %arg12) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %205 = "mhlo.get_tuple_element"(%204) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %206 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %207 = call @aten.expand.83(%206) : (tensor<f32>) -> tensor<2x128x128xf32>
    %208 = call @aten.mul.112(%205, %207) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %209 = call @aten.add.125(%202, %208) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %210 = "mhlo.get_tuple_element"(%195) {index = 1 : i32} : (tuple<tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>>) -> tensor<2x2x128x64xf32>
    %211 = "mhlo.custom_call"(%210, %19, %arg10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>
    %212 = "mhlo.get_tuple_element"(%211) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<2x128x128xf32>
    %213 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %214 = call @aten.expand.83(%213) : (tensor<f32>) -> tensor<2x128x128xf32>
    %215 = call @aten.mul.112(%212, %214) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %216 = call @aten.add.125(%209, %215) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %217 = "mhlo.get_tuple_element"(%18) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %218 = "mhlo.get_tuple_element"(%18) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %219 = "mhlo.get_tuple_element"(%18) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>>) -> tensor<256xf32>
    %220 = "mhlo.custom_call"(%216, %217, %arg6, %218, %219) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>
    %221 = "mhlo.get_tuple_element"(%220) {index = 0 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %222 = call @aten.view.212(%221) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %223 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %224 = call @aten.expand.245(%223) : (tensor<f32>) -> tensor<256x128xf32>
    %225 = call @aten.where.418(%90, %222, %224) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %226 = call @aten.index_put.461(%78, %86, %225) : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %227 = call @aten.permute.468(%226) : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %228 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %229 = call @aten.expand.237(%228) : (tensor<f32>) -> tensor<30522x128xf32>
    %230 = call @aten.mul.472(%227, %229) : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %231 = call @aten.add.491(%76, %230) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<30522x128xf32>, tensor<30522x128xf32>) -> tensor<30522x128xf32>
    %232 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %233 = call @aten.expand.513(%232) : (tensor<f32>) -> tensor<2x128xf32>
    %234 = call @aten.view.98(%5) : (tensor<2x128xi64>) -> tensor<256xi64>
    %235 = mhlo.constant dense<0> : tensor<i64>
    %236 = call @aten.lt.438(%234, %235) : (tensor<256xi64>, tensor<i64>) -> tensor<256xi1>
    %237 = mhlo.constant dense<2> : tensor<i64>
    %238 = call @aten.expand.425(%237) : (tensor<i64>) -> tensor<256xi64>
    %239 = call @aten.add.432(%234, %238) : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %240 = call @aten.where.444(%236, %239, %234) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    %241 = call @aten.stack.450(%240) : (tensor<256xi64>) -> tensor<256x1xi64>
    %242 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %243 = call @aten.ne.401(%234, %242) : (tensor<256xi64>, tensor<f64>) -> tensor<256xi1>
    %244 = call @aten.view.408(%243) : (tensor<256xi1>) -> tensor<256x1xi1>
    %245 = call @aten.expand.412(%244) : (tensor<256x1xi1>) -> tensor<256x128xi1>
    %246 = call @aten.view.212(%221) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %247 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %248 = call @aten.expand.245(%247) : (tensor<f32>) -> tensor<256x128xf32>
    %249 = call @aten.where.418(%245, %246, %248) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    %250 = call @aten.index_put.524(%233, %241, %249) : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %251 = call @aten.permute.531(%250) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %252 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %253 = call @aten.expand.617(%252) : (tensor<f32>) -> tensor<512x128xf32>
    %254 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 512]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x512xi64>
    %255 = "mhlo.slice"(%254) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %256 = call @aten.view.68(%255) : (tensor<1x128xi64>) -> tensor<128xi64>
    %257 = mhlo.constant dense<0> : tensor<i64>
    %258 = call @aten.lt.599(%256, %257) : (tensor<128xi64>, tensor<i64>) -> tensor<128xi1>
    %259 = mhlo.constant dense<512> : tensor<i64>
    %260 = call @aten.expand.586(%259) : (tensor<i64>) -> tensor<128xi64>
    %261 = call @aten.add.593(%256, %260) : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %262 = call @aten.where.605(%258, %261, %256) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    %263 = call @aten.stack.611(%262) : (tensor<128xi64>) -> tensor<128x1xi64>
    %264 = mhlo.constant dense<-1.000000e+00> : tensor<f64>
    %265 = call @aten.ne.562(%256, %264) : (tensor<128xi64>, tensor<f64>) -> tensor<128xi1>
    %266 = call @aten.view.569(%265) : (tensor<128xi1>) -> tensor<128x1xi1>
    %267 = call @aten.expand.573(%266) : (tensor<128x1xi1>) -> tensor<128x128xi1>
    %268 = "mhlo.get_tuple_element"(%220) {index = 3 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<2x128x128xf32>
    %269 = call @aten.sum.547(%268) : (tensor<2x128x128xf32>) -> tensor<1x128x128xf32>
    %270 = call @aten.view.554(%269) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    %271 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %272 = call @aten.expand.536(%271) : (tensor<f32>) -> tensor<128x128xf32>
    %273 = call @aten.where.579(%267, %270, %272) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %274 = call @aten.index_put.628(%253, %263, %273) : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %275 = call @aten.permute.635(%274) : (tensor<512x128xf32>) -> tensor<512x128xf32>
    %276 = "mhlo.get_tuple_element"(%220) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %277 = "mhlo.get_tuple_element"(%220) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %278 = "mhlo.get_tuple_element"(%197) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %279 = "mhlo.get_tuple_element"(%197) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %280 = "mhlo.get_tuple_element"(%211) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %281 = "mhlo.get_tuple_element"(%211) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %282 = "mhlo.get_tuple_element"(%204) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %283 = "mhlo.get_tuple_element"(%204) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %284 = "mhlo.get_tuple_element"(%186) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %285 = "mhlo.get_tuple_element"(%186) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %286 = "mhlo.get_tuple_element"(%181) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %287 = "mhlo.get_tuple_element"(%181) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %288 = "mhlo.get_tuple_element"(%172) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %289 = "mhlo.get_tuple_element"(%172) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %290 = "mhlo.get_tuple_element"(%168) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %291 = "mhlo.get_tuple_element"(%168) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %292 = "mhlo.get_tuple_element"(%163) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %293 = "mhlo.get_tuple_element"(%163) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %294 = "mhlo.get_tuple_element"(%140) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %295 = "mhlo.get_tuple_element"(%140) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %296 = "mhlo.get_tuple_element"(%154) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %297 = "mhlo.get_tuple_element"(%154) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %298 = "mhlo.get_tuple_element"(%147) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %299 = "mhlo.get_tuple_element"(%147) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %300 = "mhlo.get_tuple_element"(%129) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %301 = "mhlo.get_tuple_element"(%129) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %302 = "mhlo.get_tuple_element"(%124) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %303 = "mhlo.get_tuple_element"(%124) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %304 = "mhlo.get_tuple_element"(%115) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512x128xf32>
    %305 = "mhlo.get_tuple_element"(%115) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %306 = "mhlo.get_tuple_element"(%111) {index = 1 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128x512xf32>
    %307 = "mhlo.get_tuple_element"(%111) {index = 2 : i32} : (tuple<tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %308 = "mhlo.get_tuple_element"(%106) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %309 = "mhlo.get_tuple_element"(%106) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>>) -> tensor<128xf32>
    %310 = "mhlo.get_tuple_element"(%101) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128x128xf32>
    %311 = "mhlo.get_tuple_element"(%101) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %312 = "mhlo.get_tuple_element"(%97) {index = 1 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %313 = "mhlo.get_tuple_element"(%97) {index = 2 : i32} : (tuple<tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %314 = call @aten.sum.643(%arg47) : (tensor<2x128x30522xf32>) -> tensor<1x1x30522xf32>
    %315 = call @aten.view.650(%314) : (tensor<1x1x30522xf32>) -> tensor<30522xf32>
    %316 = "mhlo.tuple"(%71, %231, %251, %275, %276, %277, %278, %279, %280, %281, %282, %283, %284, %285, %286, %287, %288, %289, %290, %291, %292, %293, %294, %295, %296, %297, %298, %299, %300, %301, %302, %303, %304, %305, %306, %307, %308, %309, %310, %311, %312, %313, %315) : (tensor<2x128x30522xf32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) -> !tuple
    return %316 : !tuple
  }
  func private @aten.view.98(%arg0: tensor<2x128xi64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.index_select.118(%arg0: tensor<30522x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func private @aten.view.108(%arg0: tensor<256x128xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.expand.92(%arg0: tensor<1x128xi64>) -> tensor<2x128xi64> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xi64>) -> tensor<1x128xi64>
    %1 = "mhlo.reshape"(%0) : (tensor<1x128xi64>) -> tensor<128xi64>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    return %2 : tensor<2x128xi64>
  }
  func private @aten.index_select.102(%arg0: tensor<2x128xf32>, %arg1: tensor<256xi64>) -> tensor<256x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<256xi64>) -> tensor<256xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
  func private @aten.expand.83(%arg0: tensor<f32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func private @aten.mul.112(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.add.125(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.view.68(%arg0: tensor<1x128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.index_select.72(%arg0: tensor<512x128xf32>, %arg1: tensor<128xi64>) -> tensor<128x128xf32> {
    %0 = "mhlo.convert"(%arg1) : (tensor<128xi64>) -> tensor<128xui32>
    %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
  }
  func private @aten.view.78(%arg0: tensor<128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
  func private @aten.view.145(%arg0: tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func private @aten.view.212(%arg0: tensor<2x128x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func private @aten.permute.62(%arg0: tensor<30522x128xf32>) -> tensor<128x30522xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    return %0 : tensor<128x30522xf32>
  }
  func private @aten.mm.216(%arg0: tensor<256x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<256x30522xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x128xf32>, tensor<128x30522xf32>) -> tensor<256x30522xf32>
    return %0 : tensor<256x30522xf32>
  }
  func private @aten.view.221(%arg0: tensor<256x30522xf32>) -> tensor<2x128x30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    return %0 : tensor<2x128x30522xf32>
  }
  func private @aten.expand.50(%arg0: tensor<f32>) -> tensor<30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<30522xf32>
    return %3 : tensor<30522xf32>
  }
  func private @aten.mul.57(%arg0: tensor<30522xf32>, %arg1: tensor<30522xf32>) -> tensor<30522xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<30522xf32>
    return %0 : tensor<30522xf32>
  }
  func private @aten.add.225(%arg0: tensor<2x128x30522xf32>, %arg1: tensor<30522xf32>) -> tensor<2x128x30522xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %1 = mhlo.add %arg0, %0 : tensor<2x128x30522xf32>
    return %1 : tensor<2x128x30522xf32>
  }
  func private @aten.view.231(%arg0: tensor<2x128x30522xf32>) -> tensor<256x30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    return %0 : tensor<256x30522xf32>
  }
  func private @aten.permute.478(%arg0: tensor<256x128xf32>) -> tensor<128x256xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<128x256xf32>
    return %0 : tensor<128x256xf32>
  }
  func private @aten.mm.482(%arg0: tensor<128x256xf32>, %arg1: tensor<256x30522xf32>) -> tensor<128x30522xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<128x256xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    return %0 : tensor<128x30522xf32>
  }
  func private @aten.permute.487(%arg0: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.expand.237(%arg0: tensor<f32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<30522x128xf32>
    return %3 : tensor<30522x128xf32>
  }
  func private @aten.lt.438(%arg0: tensor<256xi64>, %arg1: tensor<i64>) -> tensor<256xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    return %1 : tensor<256xi1>
  }
  func private @aten.expand.425(%arg0: tensor<i64>) -> tensor<256xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<256xi64>
    return %3 : tensor<256xi64>
  }
  func private @aten.add.432(%arg0: tensor<256xi64>, %arg1: tensor<256xi64>) -> tensor<256xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.where.444(%arg0: tensor<256xi1>, %arg1: tensor<256xi64>, %arg2: tensor<256xi64>) -> tensor<256xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256xi1>, tensor<256xi64>, tensor<256xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  func private @aten.stack.450(%arg0: tensor<256xi64>) -> tensor<256x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi64>) -> tensor<256x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<256x1xi64>) -> tensor<256x1xi64>
    return %1 : tensor<256x1xi64>
  }
  func private @aten.ne.401(%arg0: tensor<256xi64>, %arg1: tensor<f64>) -> tensor<256xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<256xi64>) -> tensor<256xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<256xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %2 : tensor<256xi1>
  }
  func private @aten.view.408(%arg0: tensor<256xi1>) -> tensor<256x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<256xi1>) -> tensor<256x1xi1>
    return %0 : tensor<256x1xi1>
  }
  func private @aten.expand.412(%arg0: tensor<256x1xi1>) -> tensor<256x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x1xi1>) -> tensor<256x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<256x1xi1>) -> tensor<256xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    return %2 : tensor<256x128xi1>
  }
  func private @aten.permute.261(%arg0: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.mm.266(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func private @aten.view.304(%arg0: tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    return %0 : tensor<2x128x2x64xf32>
  }
  func private @aten.expand.245(%arg0: tensor<f32>) -> tensor<256x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x128xf32>
    return %3 : tensor<256x128xf32>
  }
  func private @aten.where.418(%arg0: tensor<256x128xi1>, %arg1: tensor<256x128xf32>, %arg2: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<256x128xi1>, tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>
  }
  func private @aten.index_put.461(%arg0: tensor<30522x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func private @aten.permute.468(%arg0: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.mul.472(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.add.491(%arg0: tensor<30522x128xf32>, %arg1: tensor<30522x128xf32>) -> tensor<30522x128xf32> {
    %0 = mhlo.add %arg0, %arg1 {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : tensor<30522x128xf32>
    return %0 : tensor<30522x128xf32>
  }
  func private @aten.expand.513(%arg0: tensor<f32>) -> tensor<2x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x128xf32>
    return %3 : tensor<2x128xf32>
  }
  func private @aten.index_put.524(%arg0: tensor<2x128xf32>, %arg1: tensor<256x1xi64>, %arg2: tensor<256x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    return %1 : tensor<2x128xf32>
  }
  func private @aten.permute.531(%arg0: tensor<2x128xf32>) -> tensor<2x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128xf32>) -> tensor<2x128xf32>
    return %0 : tensor<2x128xf32>
  }
  func private @aten.expand.617(%arg0: tensor<f32>) -> tensor<512x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512x128xf32>
    return %3 : tensor<512x128xf32>
  }
  func private @aten.lt.599(%arg0: tensor<128xi64>, %arg1: tensor<i64>) -> tensor<128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    return %1 : tensor<128xi1>
  }
  func private @aten.expand.586(%arg0: tensor<i64>) -> tensor<128xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<i64>) -> tensor<1xi64>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = "mhlo.reshape"(%1) : (tensor<1xi64>) -> tensor<i64>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> tensor<128xi64>
    return %3 : tensor<128xi64>
  }
  func private @aten.add.593(%arg0: tensor<128xi64>, %arg1: tensor<128xi64>) -> tensor<128xi64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.where.605(%arg0: tensor<128xi1>, %arg1: tensor<128xi64>, %arg2: tensor<128xi64>) -> tensor<128xi64> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128xi1>, tensor<128xi64>, tensor<128xi64>) -> tensor<128xi64>
    return %0 : tensor<128xi64>
  }
  func private @aten.stack.611(%arg0: tensor<128xi64>) -> tensor<128x1xi64> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi64>) -> tensor<128x1xi64>
    %1 = "mhlo.concatenate"(%0) {dimension = 1 : i64} : (tensor<128x1xi64>) -> tensor<128x1xi64>
    return %1 : tensor<128x1xi64>
  }
  func private @aten.ne.562(%arg0: tensor<128xi64>, %arg1: tensor<f64>) -> tensor<128xi1> {
    %0 = "mhlo.convert"(%arg0) : (tensor<128xi64>) -> tensor<128xf64>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<128xf64>
    %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %2 : tensor<128xi1>
  }
  func private @aten.view.569(%arg0: tensor<128xi1>) -> tensor<128x1xi1> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<128xi1>) -> tensor<128x1xi1>
    return %0 : tensor<128x1xi1>
  }
  func private @aten.expand.573(%arg0: tensor<128x1xi1>) -> tensor<128x128xi1> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x1xi1>) -> tensor<128x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<128x1xi1>) -> tensor<128xi1>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    return %2 : tensor<128x128xi1>
  }
  func private @aten.sum.547(%arg0: tensor<2x128x128xf32>) -> tensor<1x128x128xf32> {
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
  func private @aten.view.554(%arg0: tensor<1x128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func private @aten.expand.536(%arg0: tensor<f32>) -> tensor<128x128xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f32>) -> tensor<1x1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1xf32>) -> tensor<f32>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128x128xf32>
    return %3 : tensor<128x128xf32>
  }
  func private @aten.where.579(%arg0: tensor<128x128xi1>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<128x128xi1>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
  func private @aten.index_put.628(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %1 = "mhlo.scatter"(%arg0, %arg1, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    return %1 : tensor<512x128xf32>
  }
  func private @aten.permute.635(%arg0: tensor<512x128xf32>) -> tensor<512x128xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<512x128xf32>
    return %0 : tensor<512x128xf32>
  }
  func private @aten.sum.643(%arg0: tensor<2x128x30522xf32>) -> tensor<1x1x30522xf32> {
    %0 = mhlo.constant dense<256> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = "mhlo.reduce"(%arg0, %1) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %4 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%4) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<30522xf32>) -> tensor<1x1x30522xf32>
    return %3 : tensor<1x1x30522xf32>
  }
  func private @aten.view.650(%arg0: tensor<1x1x30522xf32>) -> tensor<30522xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x1x30522xf32>) -> tensor<30522xf32>
    return %0 : tensor<30522xf32>
  }
}
