// RUN: byteir-opt %s -expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse -fuse-element="attach-tag=byre_elementwise_fusion" -fusion-outlining -cse | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>>
module {
  func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<32x3x224x224xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64x64x3x3xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64x64x3x3xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<128x64x3x3xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128x64x1x1xf32>, %arg41: tensor<128x128x3x3xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x128x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x3x3xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256x256x3x3xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256x128x1x1xf32>, %arg66: tensor<256x256x3x3xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256x256x3x3xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<512x256x3x3xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512x256x1x1xf32>, %arg91: tensor<512x512x3x3xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512x512x3x3xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<1000x512xf32>, %arg102: tensor<32x1000xf16>, %arg103: tensor<1000xf32>) -> !tuple {
    %0 = mhlo.constant dense<1> : tensor<i64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %3 = mhlo.constant dense<2.040100e-02> : tensor<32x512xf16>
    %4 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %6 = mhlo.constant dense<0.000000e+00> : tensor<32x64x112x112xf16>
    %7 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %8 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %9 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %10 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %11 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %12 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %13 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %14 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %15 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %16 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %17 = mhlo.constant dense<4.900000e+01> : tensor<32x512x7x7xf16>
    %18 = mhlo.constant dense<0xFC00> : tensor<f16>
    %19 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %20 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %21 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %22 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %23 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %24 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %25 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %26 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %27 = "mhlo.convert"(%arg1) : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
    %28 = "mhlo.convert"(%arg0) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %29 = mhlo.convolution(%27, %28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16>
    %30 = "mhlo.convert"(%29) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %31 = "mhlo.batch_norm_training"(%30, %arg5, %arg4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>
    %32 = "mhlo.get_tuple_element"(%31) {index = 0 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x112x112xf32>
    %33 = "mhlo.convert"(%32) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %34 = "mhlo.get_tuple_element"(%31) {index = 1 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %35 = "mhlo.get_tuple_element"(%31) {index = 2 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %36 = mhlo.add %35, %5 : tensor<64xf32>
    %37 = "mhlo.rsqrt"(%36) : (tensor<64xf32>) -> tensor<64xf32>
    %38 = mhlo.maximum %33, %6 : tensor<32x64x112x112xf16>
    %39 = "mhlo.pad"(%38, %18) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
    %40 = "mhlo.reduce_window"(%39, %18) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %727 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%727) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
    %41 = "mhlo.convert"(%arg101) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %42 = "mhlo.dot"(%arg102, %41) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x1000xf16>, tensor<1000x512xf16>) -> tensor<32x512xf16>
    %43 = "mhlo.broadcast_in_dim"(%42) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<32x512xf16>) -> tensor<32x512x7x7xf16>
    %44 = mhlo.divide %43, %17 : tensor<32x512x7x7xf16>
    %45 = "mhlo.convert"(%arg6) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %46 = mhlo.convolution(%40, %45) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %47 = "mhlo.convert"(%46) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %48 = "mhlo.batch_norm_training"(%47, %arg10, %arg9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %49 = "mhlo.get_tuple_element"(%48) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %50 = "mhlo.convert"(%49) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %51 = "mhlo.get_tuple_element"(%48) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %52 = "mhlo.get_tuple_element"(%48) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %53 = mhlo.add %52, %5 : tensor<64xf32>
    %54 = "mhlo.rsqrt"(%53) : (tensor<64xf32>) -> tensor<64xf32>
    %55 = mhlo.maximum %50, %7 : tensor<32x64x56x56xf16>
    %56 = "mhlo.convert"(%arg11) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %57 = mhlo.convolution(%55, %56) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %58 = "mhlo.convert"(%57) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %59 = "mhlo.batch_norm_training"(%58, %arg15, %arg14) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %60 = "mhlo.get_tuple_element"(%59) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %61 = "mhlo.convert"(%60) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %62 = "mhlo.get_tuple_element"(%59) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %63 = "mhlo.get_tuple_element"(%59) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %64 = mhlo.add %63, %5 : tensor<64xf32>
    %65 = "mhlo.rsqrt"(%64) : (tensor<64xf32>) -> tensor<64xf32>
    %66 = mhlo.add %61, %40 : tensor<32x64x56x56xf16>
    %67 = mhlo.maximum %66, %7 : tensor<32x64x56x56xf16>
    %68 = "mhlo.convert"(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %69 = mhlo.convolution(%67, %68) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %70 = "mhlo.convert"(%69) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %71 = "mhlo.batch_norm_training"(%70, %arg20, %arg19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %72 = "mhlo.get_tuple_element"(%71) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %73 = "mhlo.convert"(%72) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %74 = "mhlo.get_tuple_element"(%71) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %75 = "mhlo.get_tuple_element"(%71) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %76 = mhlo.add %75, %5 : tensor<64xf32>
    %77 = "mhlo.rsqrt"(%76) : (tensor<64xf32>) -> tensor<64xf32>
    %78 = mhlo.maximum %73, %7 : tensor<32x64x56x56xf16>
    %79 = "mhlo.convert"(%arg21) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %80 = mhlo.convolution(%78, %79) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %81 = "mhlo.convert"(%80) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %82 = "mhlo.batch_norm_training"(%81, %arg25, %arg24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %83 = "mhlo.get_tuple_element"(%82) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %84 = "mhlo.convert"(%83) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %85 = "mhlo.get_tuple_element"(%82) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %86 = "mhlo.get_tuple_element"(%82) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %87 = mhlo.add %86, %5 : tensor<64xf32>
    %88 = "mhlo.rsqrt"(%87) : (tensor<64xf32>) -> tensor<64xf32>
    %89 = mhlo.add %84, %67 : tensor<32x64x56x56xf16>
    %90 = mhlo.maximum %89, %7 : tensor<32x64x56x56xf16>
    %91 = "mhlo.convert"(%arg26) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %92 = mhlo.convolution(%90, %91) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16>
    %93 = "mhlo.convert"(%92) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %94 = "mhlo.batch_norm_training"(%93, %arg30, %arg29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %95 = "mhlo.get_tuple_element"(%94) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %96 = "mhlo.convert"(%95) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %97 = "mhlo.get_tuple_element"(%94) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %98 = "mhlo.get_tuple_element"(%94) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %99 = mhlo.add %98, %9 : tensor<128xf32>
    %100 = "mhlo.rsqrt"(%99) : (tensor<128xf32>) -> tensor<128xf32>
    %101 = mhlo.maximum %96, %10 : tensor<32x128x28x28xf16>
    %102 = "mhlo.convert"(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %103 = mhlo.convolution(%101, %102) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %104 = "mhlo.convert"(%103) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %105 = "mhlo.batch_norm_training"(%104, %arg35, %arg34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %106 = "mhlo.get_tuple_element"(%105) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %107 = "mhlo.convert"(%106) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %108 = "mhlo.get_tuple_element"(%105) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %109 = "mhlo.get_tuple_element"(%105) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %110 = mhlo.add %109, %9 : tensor<128xf32>
    %111 = "mhlo.rsqrt"(%110) : (tensor<128xf32>) -> tensor<128xf32>
    %112 = "mhlo.convert"(%arg40) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %113 = mhlo.convolution(%90, %112) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16>
    %114 = "mhlo.convert"(%113) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %115 = "mhlo.batch_norm_training"(%114, %arg39, %arg38) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %116 = "mhlo.get_tuple_element"(%115) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %117 = "mhlo.convert"(%116) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %118 = "mhlo.get_tuple_element"(%115) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %119 = "mhlo.get_tuple_element"(%115) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %120 = mhlo.add %119, %9 : tensor<128xf32>
    %121 = "mhlo.rsqrt"(%120) : (tensor<128xf32>) -> tensor<128xf32>
    %122 = mhlo.add %107, %117 : tensor<32x128x28x28xf16>
    %123 = mhlo.maximum %122, %10 : tensor<32x128x28x28xf16>
    %124 = "mhlo.convert"(%arg41) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %125 = mhlo.convolution(%123, %124) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %126 = "mhlo.convert"(%125) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %127 = "mhlo.batch_norm_training"(%126, %arg45, %arg44) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %128 = "mhlo.get_tuple_element"(%127) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %129 = "mhlo.convert"(%128) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %130 = "mhlo.get_tuple_element"(%127) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %131 = "mhlo.get_tuple_element"(%127) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %132 = mhlo.add %131, %9 : tensor<128xf32>
    %133 = "mhlo.rsqrt"(%132) : (tensor<128xf32>) -> tensor<128xf32>
    %134 = mhlo.maximum %129, %10 : tensor<32x128x28x28xf16>
    %135 = "mhlo.convert"(%arg46) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %136 = mhlo.convolution(%134, %135) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %137 = "mhlo.convert"(%136) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %138 = "mhlo.batch_norm_training"(%137, %arg50, %arg49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %139 = "mhlo.get_tuple_element"(%138) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %140 = "mhlo.convert"(%139) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %141 = "mhlo.get_tuple_element"(%138) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %142 = "mhlo.get_tuple_element"(%138) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %143 = mhlo.add %142, %9 : tensor<128xf32>
    %144 = "mhlo.rsqrt"(%143) : (tensor<128xf32>) -> tensor<128xf32>
    %145 = mhlo.add %140, %123 : tensor<32x128x28x28xf16>
    %146 = mhlo.maximum %145, %10 : tensor<32x128x28x28xf16>
    %147 = "mhlo.convert"(%arg51) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %148 = mhlo.convolution(%146, %147) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16>
    %149 = "mhlo.convert"(%148) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %150 = "mhlo.batch_norm_training"(%149, %arg55, %arg54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %151 = "mhlo.get_tuple_element"(%150) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %152 = "mhlo.convert"(%151) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %153 = "mhlo.get_tuple_element"(%150) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %154 = "mhlo.get_tuple_element"(%150) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %155 = mhlo.add %154, %12 : tensor<256xf32>
    %156 = "mhlo.rsqrt"(%155) : (tensor<256xf32>) -> tensor<256xf32>
    %157 = mhlo.maximum %152, %13 : tensor<32x256x14x14xf16>
    %158 = "mhlo.convert"(%arg56) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %159 = mhlo.convolution(%157, %158) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %160 = "mhlo.convert"(%159) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %161 = "mhlo.batch_norm_training"(%160, %arg60, %arg59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %162 = "mhlo.get_tuple_element"(%161) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %163 = "mhlo.convert"(%162) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %164 = "mhlo.get_tuple_element"(%161) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %165 = "mhlo.get_tuple_element"(%161) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %166 = mhlo.add %165, %12 : tensor<256xf32>
    %167 = "mhlo.rsqrt"(%166) : (tensor<256xf32>) -> tensor<256xf32>
    %168 = "mhlo.convert"(%arg65) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %169 = mhlo.convolution(%146, %168) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16>
    %170 = "mhlo.convert"(%169) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %171 = "mhlo.batch_norm_training"(%170, %arg64, %arg63) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %172 = "mhlo.get_tuple_element"(%171) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %173 = "mhlo.convert"(%172) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %174 = "mhlo.get_tuple_element"(%171) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %175 = "mhlo.get_tuple_element"(%171) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %176 = mhlo.add %175, %12 : tensor<256xf32>
    %177 = "mhlo.rsqrt"(%176) : (tensor<256xf32>) -> tensor<256xf32>
    %178 = mhlo.add %163, %173 : tensor<32x256x14x14xf16>
    %179 = mhlo.maximum %178, %13 : tensor<32x256x14x14xf16>
    %180 = "mhlo.convert"(%arg66) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %181 = mhlo.convolution(%179, %180) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %182 = "mhlo.convert"(%181) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %183 = "mhlo.batch_norm_training"(%182, %arg70, %arg69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %184 = "mhlo.get_tuple_element"(%183) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %185 = "mhlo.convert"(%184) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %186 = "mhlo.get_tuple_element"(%183) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %187 = "mhlo.get_tuple_element"(%183) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %188 = mhlo.add %187, %12 : tensor<256xf32>
    %189 = "mhlo.rsqrt"(%188) : (tensor<256xf32>) -> tensor<256xf32>
    %190 = mhlo.maximum %185, %13 : tensor<32x256x14x14xf16>
    %191 = "mhlo.convert"(%arg71) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %192 = mhlo.convolution(%190, %191) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %193 = "mhlo.convert"(%192) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %194 = "mhlo.batch_norm_training"(%193, %arg75, %arg74) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %195 = "mhlo.get_tuple_element"(%194) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %196 = "mhlo.convert"(%195) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %197 = "mhlo.get_tuple_element"(%194) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %198 = "mhlo.get_tuple_element"(%194) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %199 = mhlo.add %198, %12 : tensor<256xf32>
    %200 = "mhlo.rsqrt"(%199) : (tensor<256xf32>) -> tensor<256xf32>
    %201 = mhlo.add %196, %179 : tensor<32x256x14x14xf16>
    %202 = mhlo.maximum %201, %13 : tensor<32x256x14x14xf16>
    %203 = "mhlo.convert"(%arg76) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %204 = mhlo.convolution(%202, %203) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16>
    %205 = "mhlo.convert"(%204) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %206 = "mhlo.batch_norm_training"(%205, %arg80, %arg79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %207 = "mhlo.get_tuple_element"(%206) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %208 = "mhlo.convert"(%207) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %209 = "mhlo.get_tuple_element"(%206) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %210 = "mhlo.get_tuple_element"(%206) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %211 = mhlo.add %210, %15 : tensor<512xf32>
    %212 = "mhlo.rsqrt"(%211) : (tensor<512xf32>) -> tensor<512xf32>
    %213 = mhlo.maximum %208, %16 : tensor<32x512x7x7xf16>
    %214 = "mhlo.convert"(%arg81) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %215 = mhlo.convolution(%213, %214) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %216 = "mhlo.convert"(%215) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %217 = "mhlo.batch_norm_training"(%216, %arg85, %arg84) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %218 = "mhlo.get_tuple_element"(%217) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %219 = "mhlo.convert"(%218) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %220 = "mhlo.get_tuple_element"(%217) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %221 = "mhlo.get_tuple_element"(%217) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %222 = mhlo.add %221, %15 : tensor<512xf32>
    %223 = "mhlo.rsqrt"(%222) : (tensor<512xf32>) -> tensor<512xf32>
    %224 = "mhlo.convert"(%arg90) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %225 = mhlo.convolution(%202, %224) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16>
    %226 = "mhlo.convert"(%225) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %227 = "mhlo.batch_norm_training"(%226, %arg89, %arg88) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %228 = "mhlo.get_tuple_element"(%227) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %229 = "mhlo.convert"(%228) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %230 = "mhlo.get_tuple_element"(%227) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %231 = "mhlo.get_tuple_element"(%227) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %232 = mhlo.add %231, %15 : tensor<512xf32>
    %233 = "mhlo.rsqrt"(%232) : (tensor<512xf32>) -> tensor<512xf32>
    %234 = mhlo.add %219, %229 : tensor<32x512x7x7xf16>
    %235 = mhlo.maximum %234, %16 : tensor<32x512x7x7xf16>
    %236 = "mhlo.convert"(%arg91) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %237 = mhlo.convolution(%235, %236) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %238 = "mhlo.convert"(%237) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %239 = "mhlo.batch_norm_training"(%238, %arg95, %arg94) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %240 = "mhlo.get_tuple_element"(%239) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %241 = "mhlo.convert"(%240) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %242 = "mhlo.get_tuple_element"(%239) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %243 = "mhlo.get_tuple_element"(%239) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %244 = mhlo.add %243, %15 : tensor<512xf32>
    %245 = "mhlo.rsqrt"(%244) : (tensor<512xf32>) -> tensor<512xf32>
    %246 = mhlo.maximum %241, %16 : tensor<32x512x7x7xf16>
    %247 = "mhlo.convert"(%arg96) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %248 = mhlo.convolution(%246, %247) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %249 = "mhlo.convert"(%248) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %250 = "mhlo.batch_norm_training"(%249, %arg100, %arg99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %251 = "mhlo.get_tuple_element"(%250) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %252 = "mhlo.convert"(%251) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %253 = "mhlo.get_tuple_element"(%250) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %254 = "mhlo.get_tuple_element"(%250) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %255 = mhlo.add %254, %15 : tensor<512xf32>
    %256 = "mhlo.rsqrt"(%255) : (tensor<512xf32>) -> tensor<512xf32>
    %257 = mhlo.add %252, %235 : tensor<32x512x7x7xf16>
    %258 = mhlo.maximum %257, %16 : tensor<32x512x7x7xf16>
    %259 = "mhlo.compare"(%258, %16) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %260 = "mhlo.select"(%259, %44, %16) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %261 = mhlo.divide %14, %256 : tensor<512xf32>
    %262 = mhlo.multiply %261, %261 : tensor<512xf32>
    %263 = mhlo.subtract %262, %15 : tensor<512xf32>
    %264 = "mhlo.convert"(%260) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %265 = "mhlo.batch_norm_grad"(%249, %arg100, %253, %263, %264) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %266 = "mhlo.get_tuple_element"(%265) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %267 = "mhlo.convert"(%266) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %268 = "mhlo.get_tuple_element"(%265) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %269 = "mhlo.get_tuple_element"(%265) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %270 = "mhlo.transpose"(%247) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %271 = "mhlo.reverse"(%270) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %272 = mhlo.convolution(%267, %271) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %273 = mhlo.convolution(%246, %267) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %274 = "mhlo.transpose"(%273) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %275 = "mhlo.compare"(%246, %16) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %276 = "mhlo.select"(%275, %272, %16) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %277 = mhlo.divide %14, %245 : tensor<512xf32>
    %278 = mhlo.multiply %277, %277 : tensor<512xf32>
    %279 = mhlo.subtract %278, %15 : tensor<512xf32>
    %280 = "mhlo.convert"(%276) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %281 = "mhlo.batch_norm_grad"(%238, %arg95, %242, %279, %280) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %282 = "mhlo.get_tuple_element"(%281) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %283 = "mhlo.convert"(%282) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %284 = "mhlo.get_tuple_element"(%281) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %285 = "mhlo.get_tuple_element"(%281) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %286 = "mhlo.transpose"(%236) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %287 = "mhlo.reverse"(%286) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %288 = mhlo.convolution(%283, %287) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %289 = mhlo.convolution(%235, %283) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %290 = "mhlo.transpose"(%289) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %291 = mhlo.add %260, %288 : tensor<32x512x7x7xf16>
    %292 = "mhlo.compare"(%235, %16) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %293 = "mhlo.select"(%292, %291, %16) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %294 = mhlo.divide %14, %223 : tensor<512xf32>
    %295 = mhlo.multiply %294, %294 : tensor<512xf32>
    %296 = mhlo.subtract %295, %15 : tensor<512xf32>
    %297 = "mhlo.convert"(%293) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %298 = "mhlo.batch_norm_grad"(%216, %arg85, %220, %296, %297) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %299 = "mhlo.get_tuple_element"(%298) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %300 = "mhlo.convert"(%299) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %301 = "mhlo.get_tuple_element"(%298) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %302 = "mhlo.get_tuple_element"(%298) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %303 = "mhlo.transpose"(%214) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %304 = "mhlo.reverse"(%303) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %305 = mhlo.convolution(%300, %304) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %306 = mhlo.convolution(%213, %300) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %307 = "mhlo.transpose"(%306) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %308 = "mhlo.compare"(%213, %16) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %309 = "mhlo.select"(%308, %305, %16) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %310 = mhlo.divide %14, %212 : tensor<512xf32>
    %311 = mhlo.multiply %310, %310 : tensor<512xf32>
    %312 = mhlo.subtract %311, %15 : tensor<512xf32>
    %313 = "mhlo.convert"(%309) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %314 = "mhlo.batch_norm_grad"(%205, %arg80, %209, %312, %313) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %315 = "mhlo.get_tuple_element"(%314) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %316 = "mhlo.convert"(%315) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %317 = "mhlo.get_tuple_element"(%314) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %318 = "mhlo.get_tuple_element"(%314) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %319 = "mhlo.transpose"(%203) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %320 = "mhlo.reverse"(%319) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %321 = mhlo.convolution(%316, %320) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<32x256x14x14xf16>
    %322 = mhlo.convolution(%202, %316) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %323 = "mhlo.transpose"(%322) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    %324 = mhlo.divide %14, %233 : tensor<512xf32>
    %325 = mhlo.multiply %324, %324 : tensor<512xf32>
    %326 = mhlo.subtract %325, %15 : tensor<512xf32>
    %327 = "mhlo.batch_norm_grad"(%226, %arg89, %230, %326, %297) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>
    %328 = "mhlo.get_tuple_element"(%327) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf32>
    %329 = "mhlo.convert"(%328) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %330 = "mhlo.get_tuple_element"(%327) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %331 = "mhlo.get_tuple_element"(%327) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %332 = "mhlo.transpose"(%224) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %333 = mhlo.convolution(%329, %332) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
    %334 = mhlo.convolution(%202, %329) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %335 = "mhlo.transpose"(%334) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    %336 = mhlo.add %333, %321 : tensor<32x256x14x14xf16>
    %337 = "mhlo.compare"(%202, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %338 = "mhlo.select"(%337, %336, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %339 = mhlo.divide %11, %200 : tensor<256xf32>
    %340 = mhlo.multiply %339, %339 : tensor<256xf32>
    %341 = mhlo.subtract %340, %12 : tensor<256xf32>
    %342 = "mhlo.convert"(%338) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %343 = "mhlo.batch_norm_grad"(%193, %arg75, %197, %341, %342) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %344 = "mhlo.get_tuple_element"(%343) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %345 = "mhlo.convert"(%344) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %346 = "mhlo.get_tuple_element"(%343) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %347 = "mhlo.get_tuple_element"(%343) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %348 = "mhlo.transpose"(%191) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %349 = "mhlo.reverse"(%348) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %350 = mhlo.convolution(%345, %349) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %351 = mhlo.convolution(%190, %345) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %352 = "mhlo.transpose"(%351) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %353 = "mhlo.compare"(%190, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %354 = "mhlo.select"(%353, %350, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %355 = mhlo.divide %11, %189 : tensor<256xf32>
    %356 = mhlo.multiply %355, %355 : tensor<256xf32>
    %357 = mhlo.subtract %356, %12 : tensor<256xf32>
    %358 = "mhlo.convert"(%354) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %359 = "mhlo.batch_norm_grad"(%182, %arg70, %186, %357, %358) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %360 = "mhlo.get_tuple_element"(%359) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %361 = "mhlo.convert"(%360) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %362 = "mhlo.get_tuple_element"(%359) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %363 = "mhlo.get_tuple_element"(%359) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %364 = "mhlo.transpose"(%180) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %365 = "mhlo.reverse"(%364) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %366 = mhlo.convolution(%361, %365) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %367 = mhlo.convolution(%179, %361) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %368 = "mhlo.transpose"(%367) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %369 = mhlo.add %338, %366 : tensor<32x256x14x14xf16>
    %370 = "mhlo.compare"(%179, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %371 = "mhlo.select"(%370, %369, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %372 = mhlo.divide %11, %167 : tensor<256xf32>
    %373 = mhlo.multiply %372, %372 : tensor<256xf32>
    %374 = mhlo.subtract %373, %12 : tensor<256xf32>
    %375 = "mhlo.convert"(%371) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %376 = "mhlo.batch_norm_grad"(%160, %arg60, %164, %374, %375) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %377 = "mhlo.get_tuple_element"(%376) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %378 = "mhlo.convert"(%377) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %379 = "mhlo.get_tuple_element"(%376) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %380 = "mhlo.get_tuple_element"(%376) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %381 = "mhlo.transpose"(%158) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %382 = "mhlo.reverse"(%381) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %383 = mhlo.convolution(%378, %382) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %384 = mhlo.convolution(%157, %378) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %385 = "mhlo.transpose"(%384) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %386 = "mhlo.compare"(%157, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %387 = "mhlo.select"(%386, %383, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %388 = mhlo.divide %11, %156 : tensor<256xf32>
    %389 = mhlo.multiply %388, %388 : tensor<256xf32>
    %390 = mhlo.subtract %389, %12 : tensor<256xf32>
    %391 = "mhlo.convert"(%387) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %392 = "mhlo.batch_norm_grad"(%149, %arg55, %153, %390, %391) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %393 = "mhlo.get_tuple_element"(%392) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %394 = "mhlo.convert"(%393) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %395 = "mhlo.get_tuple_element"(%392) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %396 = "mhlo.get_tuple_element"(%392) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %397 = "mhlo.transpose"(%147) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %398 = "mhlo.reverse"(%397) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %399 = mhlo.convolution(%394, %398) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<32x128x28x28xf16>
    %400 = mhlo.convolution(%146, %394) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %401 = "mhlo.transpose"(%400) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    %402 = mhlo.divide %11, %177 : tensor<256xf32>
    %403 = mhlo.multiply %402, %402 : tensor<256xf32>
    %404 = mhlo.subtract %403, %12 : tensor<256xf32>
    %405 = "mhlo.batch_norm_grad"(%170, %arg64, %174, %404, %375) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>
    %406 = "mhlo.get_tuple_element"(%405) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf32>
    %407 = "mhlo.convert"(%406) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %408 = "mhlo.get_tuple_element"(%405) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %409 = "mhlo.get_tuple_element"(%405) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %410 = "mhlo.transpose"(%168) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %411 = mhlo.convolution(%407, %410) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<32x128x28x28xf16>
    %412 = mhlo.convolution(%146, %407) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %413 = "mhlo.transpose"(%412) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    %414 = mhlo.add %411, %399 : tensor<32x128x28x28xf16>
    %415 = "mhlo.compare"(%146, %10) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %416 = "mhlo.select"(%415, %414, %10) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %417 = mhlo.divide %8, %144 : tensor<128xf32>
    %418 = mhlo.multiply %417, %417 : tensor<128xf32>
    %419 = mhlo.subtract %418, %9 : tensor<128xf32>
    %420 = "mhlo.convert"(%416) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %421 = "mhlo.batch_norm_grad"(%137, %arg50, %141, %419, %420) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %422 = "mhlo.get_tuple_element"(%421) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %423 = "mhlo.convert"(%422) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %424 = "mhlo.get_tuple_element"(%421) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %425 = "mhlo.get_tuple_element"(%421) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %426 = "mhlo.transpose"(%135) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %427 = "mhlo.reverse"(%426) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %428 = mhlo.convolution(%423, %427) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %429 = mhlo.convolution(%134, %423) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %430 = "mhlo.transpose"(%429) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %431 = "mhlo.compare"(%134, %10) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %432 = "mhlo.select"(%431, %428, %10) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %433 = mhlo.divide %8, %133 : tensor<128xf32>
    %434 = mhlo.multiply %433, %433 : tensor<128xf32>
    %435 = mhlo.subtract %434, %9 : tensor<128xf32>
    %436 = "mhlo.convert"(%432) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %437 = "mhlo.batch_norm_grad"(%126, %arg45, %130, %435, %436) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %438 = "mhlo.get_tuple_element"(%437) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %439 = "mhlo.convert"(%438) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %440 = "mhlo.get_tuple_element"(%437) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %441 = "mhlo.get_tuple_element"(%437) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %442 = "mhlo.transpose"(%124) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %443 = "mhlo.reverse"(%442) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %444 = mhlo.convolution(%439, %443) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %445 = mhlo.convolution(%123, %439) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %446 = "mhlo.transpose"(%445) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %447 = mhlo.add %416, %444 : tensor<32x128x28x28xf16>
    %448 = "mhlo.compare"(%123, %10) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %449 = "mhlo.select"(%448, %447, %10) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %450 = mhlo.divide %8, %111 : tensor<128xf32>
    %451 = mhlo.multiply %450, %450 : tensor<128xf32>
    %452 = mhlo.subtract %451, %9 : tensor<128xf32>
    %453 = "mhlo.convert"(%449) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %454 = "mhlo.batch_norm_grad"(%104, %arg35, %108, %452, %453) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %455 = "mhlo.get_tuple_element"(%454) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %456 = "mhlo.convert"(%455) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %457 = "mhlo.get_tuple_element"(%454) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %458 = "mhlo.get_tuple_element"(%454) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %459 = "mhlo.transpose"(%102) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %460 = "mhlo.reverse"(%459) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %461 = mhlo.convolution(%456, %460) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %462 = mhlo.convolution(%101, %456) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %463 = "mhlo.transpose"(%462) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %464 = "mhlo.compare"(%101, %10) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %465 = "mhlo.select"(%464, %461, %10) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %466 = mhlo.divide %8, %100 : tensor<128xf32>
    %467 = mhlo.multiply %466, %466 : tensor<128xf32>
    %468 = mhlo.subtract %467, %9 : tensor<128xf32>
    %469 = "mhlo.convert"(%465) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %470 = "mhlo.batch_norm_grad"(%93, %arg30, %97, %468, %469) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %471 = "mhlo.get_tuple_element"(%470) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %472 = "mhlo.convert"(%471) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %473 = "mhlo.get_tuple_element"(%470) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %474 = "mhlo.get_tuple_element"(%470) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %475 = "mhlo.transpose"(%91) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %476 = "mhlo.reverse"(%475) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %477 = mhlo.convolution(%472, %476) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<32x64x56x56xf16>
    %478 = mhlo.convolution(%90, %472) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %479 = "mhlo.transpose"(%478) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    %480 = mhlo.divide %8, %121 : tensor<128xf32>
    %481 = mhlo.multiply %480, %480 : tensor<128xf32>
    %482 = mhlo.subtract %481, %9 : tensor<128xf32>
    %483 = "mhlo.batch_norm_grad"(%114, %arg39, %118, %482, %453) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>
    %484 = "mhlo.get_tuple_element"(%483) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf32>
    %485 = "mhlo.convert"(%484) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %486 = "mhlo.get_tuple_element"(%483) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %487 = "mhlo.get_tuple_element"(%483) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %488 = "mhlo.transpose"(%112) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %489 = mhlo.convolution(%485, %488) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<32x64x56x56xf16>
    %490 = mhlo.convolution(%90, %485) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %491 = "mhlo.transpose"(%490) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    %492 = mhlo.add %489, %477 : tensor<32x64x56x56xf16>
    %493 = "mhlo.compare"(%90, %7) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %494 = "mhlo.select"(%493, %492, %7) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %495 = mhlo.divide %4, %88 : tensor<64xf32>
    %496 = mhlo.multiply %495, %495 : tensor<64xf32>
    %497 = mhlo.subtract %496, %5 : tensor<64xf32>
    %498 = "mhlo.convert"(%494) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %499 = "mhlo.batch_norm_grad"(%81, %arg25, %85, %497, %498) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %500 = "mhlo.get_tuple_element"(%499) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %501 = "mhlo.convert"(%500) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %502 = "mhlo.get_tuple_element"(%499) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %503 = "mhlo.get_tuple_element"(%499) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %504 = "mhlo.transpose"(%79) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %505 = "mhlo.reverse"(%504) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %506 = mhlo.convolution(%501, %505) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %507 = mhlo.convolution(%78, %501) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %508 = "mhlo.transpose"(%507) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %509 = "mhlo.compare"(%78, %7) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %510 = "mhlo.select"(%509, %506, %7) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %511 = mhlo.divide %4, %77 : tensor<64xf32>
    %512 = mhlo.multiply %511, %511 : tensor<64xf32>
    %513 = mhlo.subtract %512, %5 : tensor<64xf32>
    %514 = "mhlo.convert"(%510) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %515 = "mhlo.batch_norm_grad"(%70, %arg20, %74, %513, %514) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %516 = "mhlo.get_tuple_element"(%515) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %517 = "mhlo.convert"(%516) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %518 = "mhlo.get_tuple_element"(%515) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %519 = "mhlo.get_tuple_element"(%515) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %520 = "mhlo.transpose"(%68) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %521 = "mhlo.reverse"(%520) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %522 = mhlo.convolution(%517, %521) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %523 = mhlo.convolution(%67, %517) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %524 = "mhlo.transpose"(%523) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %525 = mhlo.add %494, %522 : tensor<32x64x56x56xf16>
    %526 = "mhlo.compare"(%67, %7) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %527 = "mhlo.select"(%526, %525, %7) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %528 = mhlo.divide %4, %65 : tensor<64xf32>
    %529 = mhlo.multiply %528, %528 : tensor<64xf32>
    %530 = mhlo.subtract %529, %5 : tensor<64xf32>
    %531 = "mhlo.convert"(%527) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %532 = "mhlo.batch_norm_grad"(%58, %arg15, %62, %530, %531) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %533 = "mhlo.get_tuple_element"(%532) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %534 = "mhlo.convert"(%533) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %535 = "mhlo.get_tuple_element"(%532) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %536 = "mhlo.get_tuple_element"(%532) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %537 = "mhlo.transpose"(%56) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %538 = "mhlo.reverse"(%537) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %539 = mhlo.convolution(%534, %538) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %540 = mhlo.convolution(%55, %534) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %541 = "mhlo.transpose"(%540) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %542 = "mhlo.compare"(%55, %7) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %543 = "mhlo.select"(%542, %539, %7) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %544 = mhlo.divide %4, %54 : tensor<64xf32>
    %545 = mhlo.multiply %544, %544 : tensor<64xf32>
    %546 = mhlo.subtract %545, %5 : tensor<64xf32>
    %547 = "mhlo.convert"(%543) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %548 = "mhlo.batch_norm_grad"(%47, %arg10, %51, %546, %547) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>
    %549 = "mhlo.get_tuple_element"(%548) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf32>
    %550 = "mhlo.convert"(%549) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %551 = "mhlo.get_tuple_element"(%548) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %552 = "mhlo.get_tuple_element"(%548) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %553 = "mhlo.transpose"(%45) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %554 = "mhlo.reverse"(%553) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %555 = mhlo.convolution(%550, %554) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %556 = mhlo.convolution(%40, %550) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %557 = "mhlo.transpose"(%556) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %558 = mhlo.add %527, %555 : tensor<32x64x56x56xf16>
    %559 = "mhlo.select_and_scatter"(%38, %558, %2) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %727 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%727) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %727 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%727) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
    %560 = "mhlo.compare"(%38, %6) {comparison_direction = "GT"} : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xi1>
    %561 = "mhlo.select"(%560, %559, %6) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %562 = mhlo.divide %4, %37 : tensor<64xf32>
    %563 = mhlo.multiply %562, %562 : tensor<64xf32>
    %564 = mhlo.subtract %563, %5 : tensor<64xf32>
    %565 = "mhlo.convert"(%561) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %566 = "mhlo.batch_norm_grad"(%30, %arg5, %34, %564, %565) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf32>) -> tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>
    %567 = "mhlo.get_tuple_element"(%566) {index = 0 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x112x112xf32>
    %568 = "mhlo.convert"(%567) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %569 = "mhlo.get_tuple_element"(%566) {index = 1 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %570 = "mhlo.get_tuple_element"(%566) {index = 2 : i32} : (tuple<tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %571 = mhlo.convolution(%27, %568) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %572 = "mhlo.transpose"(%571) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    %573 = "mhlo.convert"(%572) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %574 = "mhlo.convert"(%557) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %575 = "mhlo.convert"(%541) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %576 = "mhlo.convert"(%524) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %577 = "mhlo.convert"(%508) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %578 = "mhlo.convert"(%479) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %579 = "mhlo.convert"(%463) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %580 = "mhlo.convert"(%491) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %581 = "mhlo.convert"(%446) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %582 = "mhlo.convert"(%430) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %583 = "mhlo.convert"(%401) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %584 = "mhlo.convert"(%385) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %585 = "mhlo.convert"(%413) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %586 = "mhlo.convert"(%368) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %587 = "mhlo.convert"(%352) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %588 = "mhlo.convert"(%323) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %589 = "mhlo.convert"(%307) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %590 = "mhlo.convert"(%335) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %591 = "mhlo.convert"(%290) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %592 = "mhlo.convert"(%274) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %593 = mhlo.reduce(%258 init: %2) across dimensions = [3, 2] : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %727 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%727) : (tensor<f16>) -> ()
    }
    %594 = mhlo.multiply %593, %3 : tensor<32x512xf16>
    %595 = "mhlo.dot_general"(%594, %arg102) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    %596 = "mhlo.transpose"(%595) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %597 = "mhlo.convert"(%596) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %598 = "mhlo.convert"(%arg102) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    %599 = mhlo.reduce(%598 init: %1) across dimensions = [0] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %727 = mhlo.add %arg104, %arg105 : tensor<f32>
      "mhlo.return"(%727) : (tensor<f32>) -> ()
    }
    %600 = "mhlo.convert"(%599) : (tensor<1000xf32>) -> tensor<1000xf16>
    %601 = "mhlo.convert"(%600) : (tensor<1000xf16>) -> tensor<1000xf32>
    %602 = mhlo.multiply %34, %19 : tensor<64xf32>
    %603 = mhlo.multiply %arg3, %20 : tensor<64xf32>
    %604 = mhlo.add %602, %603 : tensor<64xf32>
    %605 = mhlo.multiply %35, %19 : tensor<64xf32>
    %606 = mhlo.multiply %arg2, %20 : tensor<64xf32>
    %607 = mhlo.add %605, %606 : tensor<64xf32>
    %608 = mhlo.multiply %51, %19 : tensor<64xf32>
    %609 = mhlo.multiply %arg8, %20 : tensor<64xf32>
    %610 = mhlo.add %608, %609 : tensor<64xf32>
    %611 = mhlo.multiply %52, %19 : tensor<64xf32>
    %612 = mhlo.multiply %arg7, %20 : tensor<64xf32>
    %613 = mhlo.add %611, %612 : tensor<64xf32>
    %614 = mhlo.multiply %62, %19 : tensor<64xf32>
    %615 = mhlo.multiply %arg13, %20 : tensor<64xf32>
    %616 = mhlo.add %614, %615 : tensor<64xf32>
    %617 = mhlo.multiply %63, %19 : tensor<64xf32>
    %618 = mhlo.multiply %arg12, %20 : tensor<64xf32>
    %619 = mhlo.add %617, %618 : tensor<64xf32>
    %620 = mhlo.multiply %74, %19 : tensor<64xf32>
    %621 = mhlo.multiply %arg18, %20 : tensor<64xf32>
    %622 = mhlo.add %620, %621 : tensor<64xf32>
    %623 = mhlo.multiply %75, %19 : tensor<64xf32>
    %624 = mhlo.multiply %arg17, %20 : tensor<64xf32>
    %625 = mhlo.add %623, %624 : tensor<64xf32>
    %626 = mhlo.multiply %85, %19 : tensor<64xf32>
    %627 = mhlo.multiply %arg23, %20 : tensor<64xf32>
    %628 = mhlo.add %626, %627 : tensor<64xf32>
    %629 = mhlo.multiply %86, %19 : tensor<64xf32>
    %630 = mhlo.multiply %arg22, %20 : tensor<64xf32>
    %631 = mhlo.add %629, %630 : tensor<64xf32>
    %632 = mhlo.multiply %97, %21 : tensor<128xf32>
    %633 = mhlo.multiply %arg28, %22 : tensor<128xf32>
    %634 = mhlo.add %632, %633 : tensor<128xf32>
    %635 = mhlo.multiply %98, %21 : tensor<128xf32>
    %636 = mhlo.multiply %arg27, %22 : tensor<128xf32>
    %637 = mhlo.add %635, %636 : tensor<128xf32>
    %638 = mhlo.multiply %108, %21 : tensor<128xf32>
    %639 = mhlo.multiply %arg33, %22 : tensor<128xf32>
    %640 = mhlo.add %638, %639 : tensor<128xf32>
    %641 = mhlo.multiply %109, %21 : tensor<128xf32>
    %642 = mhlo.multiply %arg32, %22 : tensor<128xf32>
    %643 = mhlo.add %641, %642 : tensor<128xf32>
    %644 = mhlo.multiply %118, %21 : tensor<128xf32>
    %645 = mhlo.multiply %arg37, %22 : tensor<128xf32>
    %646 = mhlo.add %644, %645 : tensor<128xf32>
    %647 = mhlo.multiply %119, %21 : tensor<128xf32>
    %648 = mhlo.multiply %arg36, %22 : tensor<128xf32>
    %649 = mhlo.add %647, %648 : tensor<128xf32>
    %650 = mhlo.multiply %130, %21 : tensor<128xf32>
    %651 = mhlo.multiply %arg43, %22 : tensor<128xf32>
    %652 = mhlo.add %650, %651 : tensor<128xf32>
    %653 = mhlo.multiply %131, %21 : tensor<128xf32>
    %654 = mhlo.multiply %arg42, %22 : tensor<128xf32>
    %655 = mhlo.add %653, %654 : tensor<128xf32>
    %656 = mhlo.multiply %141, %21 : tensor<128xf32>
    %657 = mhlo.multiply %arg48, %22 : tensor<128xf32>
    %658 = mhlo.add %656, %657 : tensor<128xf32>
    %659 = mhlo.multiply %142, %21 : tensor<128xf32>
    %660 = mhlo.multiply %arg47, %22 : tensor<128xf32>
    %661 = mhlo.add %659, %660 : tensor<128xf32>
    %662 = mhlo.multiply %153, %23 : tensor<256xf32>
    %663 = mhlo.multiply %arg53, %24 : tensor<256xf32>
    %664 = mhlo.add %662, %663 : tensor<256xf32>
    %665 = mhlo.multiply %154, %23 : tensor<256xf32>
    %666 = mhlo.multiply %arg52, %24 : tensor<256xf32>
    %667 = mhlo.add %665, %666 : tensor<256xf32>
    %668 = mhlo.multiply %164, %23 : tensor<256xf32>
    %669 = mhlo.multiply %arg58, %24 : tensor<256xf32>
    %670 = mhlo.add %668, %669 : tensor<256xf32>
    %671 = mhlo.multiply %165, %23 : tensor<256xf32>
    %672 = mhlo.multiply %arg57, %24 : tensor<256xf32>
    %673 = mhlo.add %671, %672 : tensor<256xf32>
    %674 = mhlo.multiply %174, %23 : tensor<256xf32>
    %675 = mhlo.multiply %arg62, %24 : tensor<256xf32>
    %676 = mhlo.add %674, %675 : tensor<256xf32>
    %677 = mhlo.multiply %175, %23 : tensor<256xf32>
    %678 = mhlo.multiply %arg61, %24 : tensor<256xf32>
    %679 = mhlo.add %677, %678 : tensor<256xf32>
    %680 = mhlo.multiply %186, %23 : tensor<256xf32>
    %681 = mhlo.multiply %arg68, %24 : tensor<256xf32>
    %682 = mhlo.add %680, %681 : tensor<256xf32>
    %683 = mhlo.multiply %187, %23 : tensor<256xf32>
    %684 = mhlo.multiply %arg67, %24 : tensor<256xf32>
    %685 = mhlo.add %683, %684 : tensor<256xf32>
    %686 = mhlo.multiply %197, %23 : tensor<256xf32>
    %687 = mhlo.multiply %arg73, %24 : tensor<256xf32>
    %688 = mhlo.add %686, %687 : tensor<256xf32>
    %689 = mhlo.multiply %198, %23 : tensor<256xf32>
    %690 = mhlo.multiply %arg72, %24 : tensor<256xf32>
    %691 = mhlo.add %689, %690 : tensor<256xf32>
    %692 = mhlo.multiply %209, %25 : tensor<512xf32>
    %693 = mhlo.multiply %arg78, %26 : tensor<512xf32>
    %694 = mhlo.add %692, %693 : tensor<512xf32>
    %695 = mhlo.multiply %210, %25 : tensor<512xf32>
    %696 = mhlo.multiply %arg77, %26 : tensor<512xf32>
    %697 = mhlo.add %695, %696 : tensor<512xf32>
    %698 = mhlo.multiply %220, %25 : tensor<512xf32>
    %699 = mhlo.multiply %arg83, %26 : tensor<512xf32>
    %700 = mhlo.add %698, %699 : tensor<512xf32>
    %701 = mhlo.multiply %221, %25 : tensor<512xf32>
    %702 = mhlo.multiply %arg82, %26 : tensor<512xf32>
    %703 = mhlo.add %701, %702 : tensor<512xf32>
    %704 = mhlo.multiply %230, %25 : tensor<512xf32>
    %705 = mhlo.multiply %arg87, %26 : tensor<512xf32>
    %706 = mhlo.add %704, %705 : tensor<512xf32>
    %707 = mhlo.multiply %231, %25 : tensor<512xf32>
    %708 = mhlo.multiply %arg86, %26 : tensor<512xf32>
    %709 = mhlo.add %707, %708 : tensor<512xf32>
    %710 = mhlo.multiply %242, %25 : tensor<512xf32>
    %711 = mhlo.multiply %arg93, %26 : tensor<512xf32>
    %712 = mhlo.add %710, %711 : tensor<512xf32>
    %713 = mhlo.multiply %243, %25 : tensor<512xf32>
    %714 = mhlo.multiply %arg92, %26 : tensor<512xf32>
    %715 = mhlo.add %713, %714 : tensor<512xf32>
    %716 = mhlo.multiply %253, %25 : tensor<512xf32>
    %717 = mhlo.multiply %arg98, %26 : tensor<512xf32>
    %718 = mhlo.add %716, %717 : tensor<512xf32>
    %719 = mhlo.multiply %254, %25 : tensor<512xf32>
    %720 = mhlo.multiply %arg97, %26 : tensor<512xf32>
    %721 = mhlo.add %719, %720 : tensor<512xf32>
    %722 = "mhlo.convert"(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %723 = "mhlo.dot_general"(%594, %41) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<1000x512xf16>) -> tensor<32x1000xf16>
    %724 = "mhlo.broadcast_in_dim"(%722) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<32x1000xf16>
    %725 = mhlo.add %723, %724 : tensor<32x1000xf16>
    %726 = "mhlo.tuple"(%573, %569, %570, %574, %551, %552, %575, %535, %536, %576, %518, %519, %577, %502, %503, %578, %473, %474, %579, %457, %458, %580, %486, %487, %581, %440, %441, %582, %424, %425, %583, %395, %396, %584, %379, %380, %585, %408, %409, %586, %362, %363, %587, %346, %347, %588, %317, %318, %589, %301, %302, %590, %330, %331, %591, %284, %285, %592, %268, %269, %597, %601, %604, %607, %0, %610, %613, %0, %616, %619, %0, %622, %625, %0, %628, %631, %0, %634, %637, %0, %640, %643, %0, %646, %649, %0, %652, %655, %0, %658, %661, %0, %664, %667, %0, %670, %673, %0, %676, %679, %0, %682, %685, %0, %688, %691, %0, %694, %697, %0, %700, %703, %0, %706, %709, %0, %712, %715, %0, %718, %721, %0, %725) : (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>) -> !tuple
    return %726 : !tuple
  }
}

