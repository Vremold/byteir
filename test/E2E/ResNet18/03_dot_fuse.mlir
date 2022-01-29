// RUN: byteir-opt %s -expand-hlo-tuples="entry-function=main" -cse -mhlo-flatten-tuple -cse -fuse-element="attach-tag=byre_elementwise_fusion" -fusion-outlining -cse | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>>
module {
  func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<32x3x224x224xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64x64x3x3xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64x64x3x3xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<128x64x3x3xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128x64x1x1xf32>, %arg41: tensor<128x128x3x3xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x128x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x3x3xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256x256x3x3xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256x128x1x1xf32>, %arg66: tensor<256x256x3x3xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256x256x3x3xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<512x256x3x3xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512x256x1x1xf32>, %arg91: tensor<512x512x3x3xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512x512x3x3xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<1000x512xf32>, %arg102: tensor<32x1000xf16>, %arg103: tensor<1000xf32>) -> !tuple {
    %0 = mhlo.constant dense<0.899999976> : tensor<512xf32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<512xf32>
    %2 = mhlo.constant dense<0.899999976> : tensor<256xf32>
    %3 = mhlo.constant dense<1.000000e-01> : tensor<256xf32>
    %4 = mhlo.constant dense<0.899999976> : tensor<128xf32>
    %5 = mhlo.constant dense<1.000000e-01> : tensor<128xf32>
    %6 = mhlo.constant dense<0.899999976> : tensor<64xf32>
    %7 = mhlo.constant dense<1.000000e-01> : tensor<64xf32>
    %8 = mhlo.constant dense<0xFC00> : tensor<f16>
    %9 = mhlo.constant dense<4.900000e+01> : tensor<32x512x7x7xf16>
    %10 = mhlo.constant dense<0.000000e+00> : tensor<32x512x7x7xf16>
    %11 = mhlo.constant dense<9.99999974E-6> : tensor<512xf32>
    %12 = mhlo.constant dense<1.000000e+00> : tensor<512xf32>
    %13 = mhlo.constant dense<0.000000e+00> : tensor<32x256x14x14xf16>
    %14 = mhlo.constant dense<9.99999974E-6> : tensor<256xf32>
    %15 = mhlo.constant dense<1.000000e+00> : tensor<256xf32>
    %16 = mhlo.constant dense<0.000000e+00> : tensor<32x128x28x28xf16>
    %17 = mhlo.constant dense<9.99999974E-6> : tensor<128xf32>
    %18 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
    %19 = mhlo.constant dense<0.000000e+00> : tensor<32x64x56x56xf16>
    %20 = mhlo.constant dense<0.000000e+00> : tensor<32x64x112x112xf16>
    %21 = mhlo.constant dense<9.99999974E-6> : tensor<64xf32>
    %22 = mhlo.constant dense<1.000000e+00> : tensor<64xf32>
    %23 = mhlo.constant dense<2.040100e-02> : tensor<32x512xf16>
    %24 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %25 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = mhlo.constant dense<1> : tensor<i64>
    %27 = "mhlo.convert"(%arg1) : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
    %28 = "mhlo.convert"(%arg0) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %29 = mhlo.convolution(%27, %28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16>
    %30 = "mhlo.convert"(%29) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %31:3 = "mhlo.batch_norm_training"(%30, %arg5, %arg4) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %32 = "mhlo.convert"(%31#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %33 = mhlo.add %31#2, %21 : tensor<64xf32>
    %34 = "mhlo.rsqrt"(%33) : (tensor<64xf32>) -> tensor<64xf32>
    %35 = mhlo.maximum %32, %20 : tensor<32x64x112x112xf16>
    %36 = "mhlo.pad"(%35, %8) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
    %37 = "mhlo.reduce_window"(%36, %8) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %607 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%607) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
    %38 = "mhlo.convert"(%arg101) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %39 = "mhlo.dot"(%arg102, %38) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x1000xf16>, tensor<1000x512xf16>) -> tensor<32x512xf16>
    %40 = "mhlo.broadcast_in_dim"(%39) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<32x512xf16>) -> tensor<32x512x7x7xf16>
    %41 = mhlo.divide %40, %9 : tensor<32x512x7x7xf16>
    %42 = "mhlo.convert"(%arg6) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %43 = mhlo.convolution(%37, %42) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %44 = "mhlo.convert"(%43) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %45:3 = "mhlo.batch_norm_training"(%44, %arg10, %arg9) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %46 = "mhlo.convert"(%45#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %47 = mhlo.add %45#2, %21 : tensor<64xf32>
    %48 = "mhlo.rsqrt"(%47) : (tensor<64xf32>) -> tensor<64xf32>
    %49 = mhlo.maximum %46, %19 : tensor<32x64x56x56xf16>
    %50 = "mhlo.convert"(%arg11) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %51 = mhlo.convolution(%49, %50) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %52 = "mhlo.convert"(%51) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %53:3 = "mhlo.batch_norm_training"(%52, %arg15, %arg14) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %54 = "mhlo.convert"(%53#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %55 = mhlo.add %53#2, %21 : tensor<64xf32>
    %56 = "mhlo.rsqrt"(%55) : (tensor<64xf32>) -> tensor<64xf32>
    %57 = mhlo.add %54, %37 : tensor<32x64x56x56xf16>
    %58 = mhlo.maximum %57, %19 : tensor<32x64x56x56xf16>
    %59 = "mhlo.convert"(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %60 = mhlo.convolution(%58, %59) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %61 = "mhlo.convert"(%60) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %62:3 = "mhlo.batch_norm_training"(%61, %arg20, %arg19) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %63 = "mhlo.convert"(%62#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %64 = mhlo.add %62#2, %21 : tensor<64xf32>
    %65 = "mhlo.rsqrt"(%64) : (tensor<64xf32>) -> tensor<64xf32>
    %66 = mhlo.maximum %63, %19 : tensor<32x64x56x56xf16>
    %67 = "mhlo.convert"(%arg21) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %68 = mhlo.convolution(%66, %67) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %69 = "mhlo.convert"(%68) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %70:3 = "mhlo.batch_norm_training"(%69, %arg25, %arg24) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %71 = "mhlo.convert"(%70#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %72 = mhlo.add %70#2, %21 : tensor<64xf32>
    %73 = "mhlo.rsqrt"(%72) : (tensor<64xf32>) -> tensor<64xf32>
    %74 = mhlo.add %71, %58 : tensor<32x64x56x56xf16>
    %75 = mhlo.maximum %74, %19 : tensor<32x64x56x56xf16>
    %76 = "mhlo.convert"(%arg26) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %77 = mhlo.convolution(%75, %76) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16>
    %78 = "mhlo.convert"(%77) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %79:3 = "mhlo.batch_norm_training"(%78, %arg30, %arg29) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %80 = "mhlo.convert"(%79#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %81 = mhlo.add %79#2, %17 : tensor<128xf32>
    %82 = "mhlo.rsqrt"(%81) : (tensor<128xf32>) -> tensor<128xf32>
    %83 = mhlo.maximum %80, %16 : tensor<32x128x28x28xf16>
    %84 = "mhlo.convert"(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %85 = mhlo.convolution(%83, %84) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %86 = "mhlo.convert"(%85) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %87:3 = "mhlo.batch_norm_training"(%86, %arg35, %arg34) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %88 = "mhlo.convert"(%87#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %89 = mhlo.add %87#2, %17 : tensor<128xf32>
    %90 = "mhlo.rsqrt"(%89) : (tensor<128xf32>) -> tensor<128xf32>
    %91 = "mhlo.convert"(%arg40) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %92 = mhlo.convolution(%75, %91) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16>
    %93 = "mhlo.convert"(%92) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %94:3 = "mhlo.batch_norm_training"(%93, %arg39, %arg38) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %95 = "mhlo.convert"(%94#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %96 = mhlo.add %94#2, %17 : tensor<128xf32>
    %97 = "mhlo.rsqrt"(%96) : (tensor<128xf32>) -> tensor<128xf32>
    %98 = mhlo.add %88, %95 : tensor<32x128x28x28xf16>
    %99 = mhlo.maximum %98, %16 : tensor<32x128x28x28xf16>
    %100 = "mhlo.convert"(%arg41) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %101 = mhlo.convolution(%99, %100) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %102 = "mhlo.convert"(%101) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %103:3 = "mhlo.batch_norm_training"(%102, %arg45, %arg44) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %104 = "mhlo.convert"(%103#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %105 = mhlo.add %103#2, %17 : tensor<128xf32>
    %106 = "mhlo.rsqrt"(%105) : (tensor<128xf32>) -> tensor<128xf32>
    %107 = mhlo.maximum %104, %16 : tensor<32x128x28x28xf16>
    %108 = "mhlo.convert"(%arg46) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %109 = mhlo.convolution(%107, %108) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %110 = "mhlo.convert"(%109) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %111:3 = "mhlo.batch_norm_training"(%110, %arg50, %arg49) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %112 = "mhlo.convert"(%111#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %113 = mhlo.add %111#2, %17 : tensor<128xf32>
    %114 = "mhlo.rsqrt"(%113) : (tensor<128xf32>) -> tensor<128xf32>
    %115 = mhlo.add %112, %99 : tensor<32x128x28x28xf16>
    %116 = mhlo.maximum %115, %16 : tensor<32x128x28x28xf16>
    %117 = "mhlo.convert"(%arg51) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %118 = mhlo.convolution(%116, %117) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16>
    %119 = "mhlo.convert"(%118) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %120:3 = "mhlo.batch_norm_training"(%119, %arg55, %arg54) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %121 = "mhlo.convert"(%120#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %122 = mhlo.add %120#2, %14 : tensor<256xf32>
    %123 = "mhlo.rsqrt"(%122) : (tensor<256xf32>) -> tensor<256xf32>
    %124 = mhlo.maximum %121, %13 : tensor<32x256x14x14xf16>
    %125 = "mhlo.convert"(%arg56) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %126 = mhlo.convolution(%124, %125) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %127 = "mhlo.convert"(%126) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %128:3 = "mhlo.batch_norm_training"(%127, %arg60, %arg59) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %129 = "mhlo.convert"(%128#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %130 = mhlo.add %128#2, %14 : tensor<256xf32>
    %131 = "mhlo.rsqrt"(%130) : (tensor<256xf32>) -> tensor<256xf32>
    %132 = "mhlo.convert"(%arg65) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %133 = mhlo.convolution(%116, %132) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16>
    %134 = "mhlo.convert"(%133) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %135:3 = "mhlo.batch_norm_training"(%134, %arg64, %arg63) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %136 = "mhlo.convert"(%135#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %137 = mhlo.add %135#2, %14 : tensor<256xf32>
    %138 = "mhlo.rsqrt"(%137) : (tensor<256xf32>) -> tensor<256xf32>
    %139 = mhlo.add %129, %136 : tensor<32x256x14x14xf16>
    %140 = mhlo.maximum %139, %13 : tensor<32x256x14x14xf16>
    %141 = "mhlo.convert"(%arg66) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %142 = mhlo.convolution(%140, %141) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %143 = "mhlo.convert"(%142) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %144:3 = "mhlo.batch_norm_training"(%143, %arg70, %arg69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %145 = "mhlo.convert"(%144#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %146 = mhlo.add %144#2, %14 : tensor<256xf32>
    %147 = "mhlo.rsqrt"(%146) : (tensor<256xf32>) -> tensor<256xf32>
    %148 = mhlo.maximum %145, %13 : tensor<32x256x14x14xf16>
    %149 = "mhlo.convert"(%arg71) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %150 = mhlo.convolution(%148, %149) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %151 = "mhlo.convert"(%150) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %152:3 = "mhlo.batch_norm_training"(%151, %arg75, %arg74) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %153 = "mhlo.convert"(%152#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %154 = mhlo.add %152#2, %14 : tensor<256xf32>
    %155 = "mhlo.rsqrt"(%154) : (tensor<256xf32>) -> tensor<256xf32>
    %156 = mhlo.add %153, %140 : tensor<32x256x14x14xf16>
    %157 = mhlo.maximum %156, %13 : tensor<32x256x14x14xf16>
    %158 = "mhlo.convert"(%arg76) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %159 = mhlo.convolution(%157, %158) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16>
    %160 = "mhlo.convert"(%159) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %161:3 = "mhlo.batch_norm_training"(%160, %arg80, %arg79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %162 = "mhlo.convert"(%161#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %163 = mhlo.add %161#2, %11 : tensor<512xf32>
    %164 = "mhlo.rsqrt"(%163) : (tensor<512xf32>) -> tensor<512xf32>
    %165 = mhlo.maximum %162, %10 : tensor<32x512x7x7xf16>
    %166 = "mhlo.convert"(%arg81) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %167 = mhlo.convolution(%165, %166) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %168 = "mhlo.convert"(%167) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %169:3 = "mhlo.batch_norm_training"(%168, %arg85, %arg84) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %170 = "mhlo.convert"(%169#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %171 = mhlo.add %169#2, %11 : tensor<512xf32>
    %172 = "mhlo.rsqrt"(%171) : (tensor<512xf32>) -> tensor<512xf32>
    %173 = "mhlo.convert"(%arg90) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %174 = mhlo.convolution(%157, %173) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16>
    %175 = "mhlo.convert"(%174) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %176:3 = "mhlo.batch_norm_training"(%175, %arg89, %arg88) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %177 = "mhlo.convert"(%176#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %178 = mhlo.add %176#2, %11 : tensor<512xf32>
    %179 = "mhlo.rsqrt"(%178) : (tensor<512xf32>) -> tensor<512xf32>
    %180 = mhlo.add %170, %177 : tensor<32x512x7x7xf16>
    %181 = mhlo.maximum %180, %10 : tensor<32x512x7x7xf16>
    %182 = "mhlo.convert"(%arg91) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %183 = mhlo.convolution(%181, %182) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %184 = "mhlo.convert"(%183) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %185:3 = "mhlo.batch_norm_training"(%184, %arg95, %arg94) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %186 = "mhlo.convert"(%185#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %187 = mhlo.add %185#2, %11 : tensor<512xf32>
    %188 = "mhlo.rsqrt"(%187) : (tensor<512xf32>) -> tensor<512xf32>
    %189 = mhlo.maximum %186, %10 : tensor<32x512x7x7xf16>
    %190 = "mhlo.convert"(%arg96) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %191 = mhlo.convolution(%189, %190) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %192 = "mhlo.convert"(%191) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %193:3 = "mhlo.batch_norm_training"(%192, %arg100, %arg99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %194 = "mhlo.convert"(%193#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %195 = mhlo.add %193#2, %11 : tensor<512xf32>
    %196 = "mhlo.rsqrt"(%195) : (tensor<512xf32>) -> tensor<512xf32>
    %197 = mhlo.add %194, %181 : tensor<32x512x7x7xf16>
    %198 = mhlo.maximum %197, %10 : tensor<32x512x7x7xf16>
    %199 = "mhlo.compare"(%198, %10) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %200 = "mhlo.select"(%199, %41, %10) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %201 = mhlo.divide %12, %196 : tensor<512xf32>
    %202 = mhlo.multiply %201, %201 : tensor<512xf32>
    %203 = mhlo.subtract %202, %11 : tensor<512xf32>
    %204 = "mhlo.convert"(%200) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %205:3 = "mhlo.batch_norm_grad"(%192, %arg100, %193#1, %203, %204) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %206 = "mhlo.convert"(%205#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %207 = "mhlo.transpose"(%190) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %208 = "mhlo.reverse"(%207) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %209 = mhlo.convolution(%206, %208) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %210 = mhlo.convolution(%189, %206) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %211 = "mhlo.transpose"(%210) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %212 = "mhlo.compare"(%189, %10) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %213 = "mhlo.select"(%212, %209, %10) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %214 = mhlo.divide %12, %188 : tensor<512xf32>
    %215 = mhlo.multiply %214, %214 : tensor<512xf32>
    %216 = mhlo.subtract %215, %11 : tensor<512xf32>
    %217 = "mhlo.convert"(%213) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %218:3 = "mhlo.batch_norm_grad"(%184, %arg95, %185#1, %216, %217) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %219 = "mhlo.convert"(%218#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %220 = "mhlo.transpose"(%182) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %221 = "mhlo.reverse"(%220) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %222 = mhlo.convolution(%219, %221) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %223 = mhlo.convolution(%181, %219) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %224 = "mhlo.transpose"(%223) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %225 = mhlo.add %200, %222 : tensor<32x512x7x7xf16>
    %226 = "mhlo.compare"(%181, %10) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %227 = "mhlo.select"(%226, %225, %10) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %228 = mhlo.divide %12, %172 : tensor<512xf32>
    %229 = mhlo.multiply %228, %228 : tensor<512xf32>
    %230 = mhlo.subtract %229, %11 : tensor<512xf32>
    %231 = "mhlo.convert"(%227) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %232:3 = "mhlo.batch_norm_grad"(%168, %arg85, %169#1, %230, %231) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %233 = "mhlo.convert"(%232#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %234 = "mhlo.transpose"(%166) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %235 = "mhlo.reverse"(%234) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %236 = mhlo.convolution(%233, %235) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %237 = mhlo.convolution(%165, %233) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %238 = "mhlo.transpose"(%237) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %239 = "mhlo.compare"(%165, %10) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %240 = "mhlo.select"(%239, %236, %10) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %241 = mhlo.divide %12, %164 : tensor<512xf32>
    %242 = mhlo.multiply %241, %241 : tensor<512xf32>
    %243 = mhlo.subtract %242, %11 : tensor<512xf32>
    %244 = "mhlo.convert"(%240) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %245:3 = "mhlo.batch_norm_grad"(%160, %arg80, %161#1, %243, %244) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %246 = "mhlo.convert"(%245#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %247 = "mhlo.transpose"(%158) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %248 = "mhlo.reverse"(%247) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %249 = mhlo.convolution(%246, %248) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<32x256x14x14xf16>
    %250 = mhlo.convolution(%157, %246) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %251 = "mhlo.transpose"(%250) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    %252 = mhlo.divide %12, %179 : tensor<512xf32>
    %253 = mhlo.multiply %252, %252 : tensor<512xf32>
    %254 = mhlo.subtract %253, %11 : tensor<512xf32>
    %255:3 = "mhlo.batch_norm_grad"(%175, %arg89, %176#1, %254, %231) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %256 = "mhlo.convert"(%255#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %257 = "mhlo.transpose"(%173) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %258 = mhlo.convolution(%256, %257) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
    %259 = mhlo.convolution(%157, %256) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %260 = "mhlo.transpose"(%259) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    %261 = mhlo.add %258, %249 : tensor<32x256x14x14xf16>
    %262 = "mhlo.compare"(%157, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %263 = "mhlo.select"(%262, %261, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %264 = mhlo.divide %15, %155 : tensor<256xf32>
    %265 = mhlo.multiply %264, %264 : tensor<256xf32>
    %266 = mhlo.subtract %265, %14 : tensor<256xf32>
    %267 = "mhlo.convert"(%263) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %268:3 = "mhlo.batch_norm_grad"(%151, %arg75, %152#1, %266, %267) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %269 = "mhlo.convert"(%268#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %270 = "mhlo.transpose"(%149) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %271 = "mhlo.reverse"(%270) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %272 = mhlo.convolution(%269, %271) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %273 = mhlo.convolution(%148, %269) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %274 = "mhlo.transpose"(%273) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %275 = "mhlo.compare"(%148, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %276 = "mhlo.select"(%275, %272, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %277 = mhlo.divide %15, %147 : tensor<256xf32>
    %278 = mhlo.multiply %277, %277 : tensor<256xf32>
    %279 = mhlo.subtract %278, %14 : tensor<256xf32>
    %280 = "mhlo.convert"(%276) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %281:3 = "mhlo.batch_norm_grad"(%143, %arg70, %144#1, %279, %280) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %282 = "mhlo.convert"(%281#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %283 = "mhlo.transpose"(%141) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %284 = "mhlo.reverse"(%283) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %285 = mhlo.convolution(%282, %284) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %286 = mhlo.convolution(%140, %282) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %287 = "mhlo.transpose"(%286) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %288 = mhlo.add %263, %285 : tensor<32x256x14x14xf16>
    %289 = "mhlo.compare"(%140, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %290 = "mhlo.select"(%289, %288, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %291 = mhlo.divide %15, %131 : tensor<256xf32>
    %292 = mhlo.multiply %291, %291 : tensor<256xf32>
    %293 = mhlo.subtract %292, %14 : tensor<256xf32>
    %294 = "mhlo.convert"(%290) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %295:3 = "mhlo.batch_norm_grad"(%127, %arg60, %128#1, %293, %294) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %296 = "mhlo.convert"(%295#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %297 = "mhlo.transpose"(%125) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %298 = "mhlo.reverse"(%297) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %299 = mhlo.convolution(%296, %298) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %300 = mhlo.convolution(%124, %296) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %301 = "mhlo.transpose"(%300) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %302 = "mhlo.compare"(%124, %13) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %303 = "mhlo.select"(%302, %299, %13) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %304 = mhlo.divide %15, %123 : tensor<256xf32>
    %305 = mhlo.multiply %304, %304 : tensor<256xf32>
    %306 = mhlo.subtract %305, %14 : tensor<256xf32>
    %307 = "mhlo.convert"(%303) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %308:3 = "mhlo.batch_norm_grad"(%119, %arg55, %120#1, %306, %307) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %309 = "mhlo.convert"(%308#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %310 = "mhlo.transpose"(%117) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %311 = "mhlo.reverse"(%310) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %312 = mhlo.convolution(%309, %311) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<32x128x28x28xf16>
    %313 = mhlo.convolution(%116, %309) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %314 = "mhlo.transpose"(%313) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    %315 = mhlo.divide %15, %138 : tensor<256xf32>
    %316 = mhlo.multiply %315, %315 : tensor<256xf32>
    %317 = mhlo.subtract %316, %14 : tensor<256xf32>
    %318:3 = "mhlo.batch_norm_grad"(%134, %arg64, %135#1, %317, %294) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %319 = "mhlo.convert"(%318#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %320 = "mhlo.transpose"(%132) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %321 = mhlo.convolution(%319, %320) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<32x128x28x28xf16>
    %322 = mhlo.convolution(%116, %319) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %323 = "mhlo.transpose"(%322) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    %324 = mhlo.add %321, %312 : tensor<32x128x28x28xf16>
    %325 = "mhlo.compare"(%116, %16) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %326 = "mhlo.select"(%325, %324, %16) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %327 = mhlo.divide %18, %114 : tensor<128xf32>
    %328 = mhlo.multiply %327, %327 : tensor<128xf32>
    %329 = mhlo.subtract %328, %17 : tensor<128xf32>
    %330 = "mhlo.convert"(%326) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %331:3 = "mhlo.batch_norm_grad"(%110, %arg50, %111#1, %329, %330) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %332 = "mhlo.convert"(%331#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %333 = "mhlo.transpose"(%108) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %334 = "mhlo.reverse"(%333) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %335 = mhlo.convolution(%332, %334) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %336 = mhlo.convolution(%107, %332) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %337 = "mhlo.transpose"(%336) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %338 = "mhlo.compare"(%107, %16) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %339 = "mhlo.select"(%338, %335, %16) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %340 = mhlo.divide %18, %106 : tensor<128xf32>
    %341 = mhlo.multiply %340, %340 : tensor<128xf32>
    %342 = mhlo.subtract %341, %17 : tensor<128xf32>
    %343 = "mhlo.convert"(%339) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %344:3 = "mhlo.batch_norm_grad"(%102, %arg45, %103#1, %342, %343) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %345 = "mhlo.convert"(%344#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %346 = "mhlo.transpose"(%100) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %347 = "mhlo.reverse"(%346) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %348 = mhlo.convolution(%345, %347) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %349 = mhlo.convolution(%99, %345) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %350 = "mhlo.transpose"(%349) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %351 = mhlo.add %326, %348 : tensor<32x128x28x28xf16>
    %352 = "mhlo.compare"(%99, %16) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %353 = "mhlo.select"(%352, %351, %16) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %354 = mhlo.divide %18, %90 : tensor<128xf32>
    %355 = mhlo.multiply %354, %354 : tensor<128xf32>
    %356 = mhlo.subtract %355, %17 : tensor<128xf32>
    %357 = "mhlo.convert"(%353) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %358:3 = "mhlo.batch_norm_grad"(%86, %arg35, %87#1, %356, %357) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %359 = "mhlo.convert"(%358#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %360 = "mhlo.transpose"(%84) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %361 = "mhlo.reverse"(%360) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %362 = mhlo.convolution(%359, %361) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %363 = mhlo.convolution(%83, %359) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %364 = "mhlo.transpose"(%363) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %365 = "mhlo.compare"(%83, %16) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %366 = "mhlo.select"(%365, %362, %16) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %367 = mhlo.divide %18, %82 : tensor<128xf32>
    %368 = mhlo.multiply %367, %367 : tensor<128xf32>
    %369 = mhlo.subtract %368, %17 : tensor<128xf32>
    %370 = "mhlo.convert"(%366) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %371:3 = "mhlo.batch_norm_grad"(%78, %arg30, %79#1, %369, %370) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %372 = "mhlo.convert"(%371#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %373 = "mhlo.transpose"(%76) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %374 = "mhlo.reverse"(%373) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %375 = mhlo.convolution(%372, %374) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<32x64x56x56xf16>
    %376 = mhlo.convolution(%75, %372) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %377 = "mhlo.transpose"(%376) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    %378 = mhlo.divide %18, %97 : tensor<128xf32>
    %379 = mhlo.multiply %378, %378 : tensor<128xf32>
    %380 = mhlo.subtract %379, %17 : tensor<128xf32>
    %381:3 = "mhlo.batch_norm_grad"(%93, %arg39, %94#1, %380, %357) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %382 = "mhlo.convert"(%381#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %383 = "mhlo.transpose"(%91) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %384 = mhlo.convolution(%382, %383) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<32x64x56x56xf16>
    %385 = mhlo.convolution(%75, %382) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %386 = "mhlo.transpose"(%385) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    %387 = mhlo.add %384, %375 : tensor<32x64x56x56xf16>
    %388 = "mhlo.compare"(%75, %19) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %389 = "mhlo.select"(%388, %387, %19) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %390 = mhlo.divide %22, %73 : tensor<64xf32>
    %391 = mhlo.multiply %390, %390 : tensor<64xf32>
    %392 = mhlo.subtract %391, %21 : tensor<64xf32>
    %393 = "mhlo.convert"(%389) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %394:3 = "mhlo.batch_norm_grad"(%69, %arg25, %70#1, %392, %393) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %395 = "mhlo.convert"(%394#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %396 = "mhlo.transpose"(%67) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %397 = "mhlo.reverse"(%396) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %398 = mhlo.convolution(%395, %397) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %399 = mhlo.convolution(%66, %395) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %400 = "mhlo.transpose"(%399) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %401 = "mhlo.compare"(%66, %19) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %402 = "mhlo.select"(%401, %398, %19) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %403 = mhlo.divide %22, %65 : tensor<64xf32>
    %404 = mhlo.multiply %403, %403 : tensor<64xf32>
    %405 = mhlo.subtract %404, %21 : tensor<64xf32>
    %406 = "mhlo.convert"(%402) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %407:3 = "mhlo.batch_norm_grad"(%61, %arg20, %62#1, %405, %406) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %408 = "mhlo.convert"(%407#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %409 = "mhlo.transpose"(%59) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %410 = "mhlo.reverse"(%409) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %411 = mhlo.convolution(%408, %410) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %412 = mhlo.convolution(%58, %408) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %413 = "mhlo.transpose"(%412) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %414 = mhlo.add %389, %411 : tensor<32x64x56x56xf16>
    %415 = "mhlo.compare"(%58, %19) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %416 = "mhlo.select"(%415, %414, %19) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %417 = mhlo.divide %22, %56 : tensor<64xf32>
    %418 = mhlo.multiply %417, %417 : tensor<64xf32>
    %419 = mhlo.subtract %418, %21 : tensor<64xf32>
    %420 = "mhlo.convert"(%416) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %421:3 = "mhlo.batch_norm_grad"(%52, %arg15, %53#1, %419, %420) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %422 = "mhlo.convert"(%421#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %423 = "mhlo.transpose"(%50) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %424 = "mhlo.reverse"(%423) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %425 = mhlo.convolution(%422, %424) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %426 = mhlo.convolution(%49, %422) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %427 = "mhlo.transpose"(%426) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %428 = "mhlo.compare"(%49, %19) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %429 = "mhlo.select"(%428, %425, %19) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %430 = mhlo.divide %22, %48 : tensor<64xf32>
    %431 = mhlo.multiply %430, %430 : tensor<64xf32>
    %432 = mhlo.subtract %431, %21 : tensor<64xf32>
    %433 = "mhlo.convert"(%429) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %434:3 = "mhlo.batch_norm_grad"(%44, %arg10, %45#1, %432, %433) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %435 = "mhlo.convert"(%434#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %436 = "mhlo.transpose"(%42) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %437 = "mhlo.reverse"(%436) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %438 = mhlo.convolution(%435, %437) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %439 = mhlo.convolution(%37, %435) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %440 = "mhlo.transpose"(%439) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %441 = mhlo.add %416, %438 : tensor<32x64x56x56xf16>
    %442 = "mhlo.select_and_scatter"(%35, %441, %24) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %607 = "mhlo.compare"(%arg104, %arg105) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%607) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %607 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%607) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
    %443 = "mhlo.compare"(%35, %20) {comparison_direction = "GT"} : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xi1>
    %444 = "mhlo.select"(%443, %442, %20) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %445 = mhlo.divide %22, %34 : tensor<64xf32>
    %446 = mhlo.multiply %445, %445 : tensor<64xf32>
    %447 = mhlo.subtract %446, %21 : tensor<64xf32>
    %448 = "mhlo.convert"(%444) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %449:3 = "mhlo.batch_norm_grad"(%30, %arg5, %31#1, %447, %448) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %450 = "mhlo.convert"(%449#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %451 = mhlo.convolution(%27, %450) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %452 = "mhlo.transpose"(%451) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    %453 = "mhlo.convert"(%452) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %454 = "mhlo.convert"(%440) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %455 = "mhlo.convert"(%427) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %456 = "mhlo.convert"(%413) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %457 = "mhlo.convert"(%400) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %458 = "mhlo.convert"(%377) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %459 = "mhlo.convert"(%364) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %460 = "mhlo.convert"(%386) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %461 = "mhlo.convert"(%350) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %462 = "mhlo.convert"(%337) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %463 = "mhlo.convert"(%314) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %464 = "mhlo.convert"(%301) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %465 = "mhlo.convert"(%323) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %466 = "mhlo.convert"(%287) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %467 = "mhlo.convert"(%274) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %468 = "mhlo.convert"(%251) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %469 = "mhlo.convert"(%238) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %470 = "mhlo.convert"(%260) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %471 = "mhlo.convert"(%224) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %472 = "mhlo.convert"(%211) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %473 = mhlo.reduce(%198 init: %24) across dimensions = [3, 2] : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512xf16>
     reducer(%arg104: tensor<f16>, %arg105: tensor<f16>)  {
      %607 = mhlo.add %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%607) : (tensor<f16>) -> ()
    }
    %474 = mhlo.multiply %473, %23 : tensor<32x512xf16>
    %475 = "mhlo.dot_general"(%474, %arg102) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    %476 = "mhlo.transpose"(%475) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %477 = "mhlo.convert"(%476) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %478 = "mhlo.convert"(%arg102) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    %479 = mhlo.reduce(%478 init: %25) across dimensions = [0] : (tensor<32x1000xf32>, tensor<f32>) -> tensor<1000xf32>
     reducer(%arg104: tensor<f32>, %arg105: tensor<f32>)  {
      %607 = mhlo.add %arg104, %arg105 : tensor<f32>
      "mhlo.return"(%607) : (tensor<f32>) -> ()
    }
    %480 = "mhlo.convert"(%479) : (tensor<1000xf32>) -> tensor<1000xf16>
    %481 = "mhlo.convert"(%480) : (tensor<1000xf16>) -> tensor<1000xf32>
    %482 = mhlo.multiply %31#1, %7 : tensor<64xf32>
    %483 = mhlo.multiply %arg3, %6 : tensor<64xf32>
    %484 = mhlo.add %482, %483 : tensor<64xf32>
    %485 = mhlo.multiply %31#2, %7 : tensor<64xf32>
    %486 = mhlo.multiply %arg2, %6 : tensor<64xf32>
    %487 = mhlo.add %485, %486 : tensor<64xf32>
    %488 = mhlo.multiply %45#1, %7 : tensor<64xf32>
    %489 = mhlo.multiply %arg8, %6 : tensor<64xf32>
    %490 = mhlo.add %488, %489 : tensor<64xf32>
    %491 = mhlo.multiply %45#2, %7 : tensor<64xf32>
    %492 = mhlo.multiply %arg7, %6 : tensor<64xf32>
    %493 = mhlo.add %491, %492 : tensor<64xf32>
    %494 = mhlo.multiply %53#1, %7 : tensor<64xf32>
    %495 = mhlo.multiply %arg13, %6 : tensor<64xf32>
    %496 = mhlo.add %494, %495 : tensor<64xf32>
    %497 = mhlo.multiply %53#2, %7 : tensor<64xf32>
    %498 = mhlo.multiply %arg12, %6 : tensor<64xf32>
    %499 = mhlo.add %497, %498 : tensor<64xf32>
    %500 = mhlo.multiply %62#1, %7 : tensor<64xf32>
    %501 = mhlo.multiply %arg18, %6 : tensor<64xf32>
    %502 = mhlo.add %500, %501 : tensor<64xf32>
    %503 = mhlo.multiply %62#2, %7 : tensor<64xf32>
    %504 = mhlo.multiply %arg17, %6 : tensor<64xf32>
    %505 = mhlo.add %503, %504 : tensor<64xf32>
    %506 = mhlo.multiply %70#1, %7 : tensor<64xf32>
    %507 = mhlo.multiply %arg23, %6 : tensor<64xf32>
    %508 = mhlo.add %506, %507 : tensor<64xf32>
    %509 = mhlo.multiply %70#2, %7 : tensor<64xf32>
    %510 = mhlo.multiply %arg22, %6 : tensor<64xf32>
    %511 = mhlo.add %509, %510 : tensor<64xf32>
    %512 = mhlo.multiply %79#1, %5 : tensor<128xf32>
    %513 = mhlo.multiply %arg28, %4 : tensor<128xf32>
    %514 = mhlo.add %512, %513 : tensor<128xf32>
    %515 = mhlo.multiply %79#2, %5 : tensor<128xf32>
    %516 = mhlo.multiply %arg27, %4 : tensor<128xf32>
    %517 = mhlo.add %515, %516 : tensor<128xf32>
    %518 = mhlo.multiply %87#1, %5 : tensor<128xf32>
    %519 = mhlo.multiply %arg33, %4 : tensor<128xf32>
    %520 = mhlo.add %518, %519 : tensor<128xf32>
    %521 = mhlo.multiply %87#2, %5 : tensor<128xf32>
    %522 = mhlo.multiply %arg32, %4 : tensor<128xf32>
    %523 = mhlo.add %521, %522 : tensor<128xf32>
    %524 = mhlo.multiply %94#1, %5 : tensor<128xf32>
    %525 = mhlo.multiply %arg37, %4 : tensor<128xf32>
    %526 = mhlo.add %524, %525 : tensor<128xf32>
    %527 = mhlo.multiply %94#2, %5 : tensor<128xf32>
    %528 = mhlo.multiply %arg36, %4 : tensor<128xf32>
    %529 = mhlo.add %527, %528 : tensor<128xf32>
    %530 = mhlo.multiply %103#1, %5 : tensor<128xf32>
    %531 = mhlo.multiply %arg43, %4 : tensor<128xf32>
    %532 = mhlo.add %530, %531 : tensor<128xf32>
    %533 = mhlo.multiply %103#2, %5 : tensor<128xf32>
    %534 = mhlo.multiply %arg42, %4 : tensor<128xf32>
    %535 = mhlo.add %533, %534 : tensor<128xf32>
    %536 = mhlo.multiply %111#1, %5 : tensor<128xf32>
    %537 = mhlo.multiply %arg48, %4 : tensor<128xf32>
    %538 = mhlo.add %536, %537 : tensor<128xf32>
    %539 = mhlo.multiply %111#2, %5 : tensor<128xf32>
    %540 = mhlo.multiply %arg47, %4 : tensor<128xf32>
    %541 = mhlo.add %539, %540 : tensor<128xf32>
    %542 = mhlo.multiply %120#1, %3 : tensor<256xf32>
    %543 = mhlo.multiply %arg53, %2 : tensor<256xf32>
    %544 = mhlo.add %542, %543 : tensor<256xf32>
    %545 = mhlo.multiply %120#2, %3 : tensor<256xf32>
    %546 = mhlo.multiply %arg52, %2 : tensor<256xf32>
    %547 = mhlo.add %545, %546 : tensor<256xf32>
    %548 = mhlo.multiply %128#1, %3 : tensor<256xf32>
    %549 = mhlo.multiply %arg58, %2 : tensor<256xf32>
    %550 = mhlo.add %548, %549 : tensor<256xf32>
    %551 = mhlo.multiply %128#2, %3 : tensor<256xf32>
    %552 = mhlo.multiply %arg57, %2 : tensor<256xf32>
    %553 = mhlo.add %551, %552 : tensor<256xf32>
    %554 = mhlo.multiply %135#1, %3 : tensor<256xf32>
    %555 = mhlo.multiply %arg62, %2 : tensor<256xf32>
    %556 = mhlo.add %554, %555 : tensor<256xf32>
    %557 = mhlo.multiply %135#2, %3 : tensor<256xf32>
    %558 = mhlo.multiply %arg61, %2 : tensor<256xf32>
    %559 = mhlo.add %557, %558 : tensor<256xf32>
    %560 = mhlo.multiply %144#1, %3 : tensor<256xf32>
    %561 = mhlo.multiply %arg68, %2 : tensor<256xf32>
    %562 = mhlo.add %560, %561 : tensor<256xf32>
    %563 = mhlo.multiply %144#2, %3 : tensor<256xf32>
    %564 = mhlo.multiply %arg67, %2 : tensor<256xf32>
    %565 = mhlo.add %563, %564 : tensor<256xf32>
    %566 = mhlo.multiply %152#1, %3 : tensor<256xf32>
    %567 = mhlo.multiply %arg73, %2 : tensor<256xf32>
    %568 = mhlo.add %566, %567 : tensor<256xf32>
    %569 = mhlo.multiply %152#2, %3 : tensor<256xf32>
    %570 = mhlo.multiply %arg72, %2 : tensor<256xf32>
    %571 = mhlo.add %569, %570 : tensor<256xf32>
    %572 = mhlo.multiply %161#1, %1 : tensor<512xf32>
    %573 = mhlo.multiply %arg78, %0 : tensor<512xf32>
    %574 = mhlo.add %572, %573 : tensor<512xf32>
    %575 = mhlo.multiply %161#2, %1 : tensor<512xf32>
    %576 = mhlo.multiply %arg77, %0 : tensor<512xf32>
    %577 = mhlo.add %575, %576 : tensor<512xf32>
    %578 = mhlo.multiply %169#1, %1 : tensor<512xf32>
    %579 = mhlo.multiply %arg83, %0 : tensor<512xf32>
    %580 = mhlo.add %578, %579 : tensor<512xf32>
    %581 = mhlo.multiply %169#2, %1 : tensor<512xf32>
    %582 = mhlo.multiply %arg82, %0 : tensor<512xf32>
    %583 = mhlo.add %581, %582 : tensor<512xf32>
    %584 = mhlo.multiply %176#1, %1 : tensor<512xf32>
    %585 = mhlo.multiply %arg87, %0 : tensor<512xf32>
    %586 = mhlo.add %584, %585 : tensor<512xf32>
    %587 = mhlo.multiply %176#2, %1 : tensor<512xf32>
    %588 = mhlo.multiply %arg86, %0 : tensor<512xf32>
    %589 = mhlo.add %587, %588 : tensor<512xf32>
    %590 = mhlo.multiply %185#1, %1 : tensor<512xf32>
    %591 = mhlo.multiply %arg93, %0 : tensor<512xf32>
    %592 = mhlo.add %590, %591 : tensor<512xf32>
    %593 = mhlo.multiply %185#2, %1 : tensor<512xf32>
    %594 = mhlo.multiply %arg92, %0 : tensor<512xf32>
    %595 = mhlo.add %593, %594 : tensor<512xf32>
    %596 = mhlo.multiply %193#1, %1 : tensor<512xf32>
    %597 = mhlo.multiply %arg98, %0 : tensor<512xf32>
    %598 = mhlo.add %596, %597 : tensor<512xf32>
    %599 = mhlo.multiply %193#2, %1 : tensor<512xf32>
    %600 = mhlo.multiply %arg97, %0 : tensor<512xf32>
    %601 = mhlo.add %599, %600 : tensor<512xf32>
    %602 = "mhlo.convert"(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %603 = "mhlo.dot_general"(%474, %38) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<1000x512xf16>) -> tensor<32x1000xf16>
    %604 = "mhlo.broadcast_in_dim"(%602) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<32x1000xf16>
    %605 = mhlo.add %603, %604 : tensor<32x1000xf16>
    %606 = "mhlo.tuple"(%453, %449#1, %449#2, %454, %434#1, %434#2, %455, %421#1, %421#2, %456, %407#1, %407#2, %457, %394#1, %394#2, %458, %371#1, %371#2, %459, %358#1, %358#2, %460, %381#1, %381#2, %461, %344#1, %344#2, %462, %331#1, %331#2, %463, %308#1, %308#2, %464, %295#1, %295#2, %465, %318#1, %318#2, %466, %281#1, %281#2, %467, %268#1, %268#2, %468, %245#1, %245#2, %469, %232#1, %232#2, %470, %255#1, %255#2, %471, %218#1, %218#2, %472, %205#1, %205#2, %477, %481, %484, %487, %26, %490, %493, %26, %496, %499, %26, %502, %505, %26, %508, %511, %26, %514, %517, %26, %520, %523, %26, %526, %529, %26, %532, %535, %26, %538, %541, %26, %544, %547, %26, %550, %553, %26, %556, %559, %26, %562, %565, %26, %568, %571, %26, %574, %577, %26, %580, %583, %26, %586, %589, %26, %592, %595, %26, %598, %601, %26, %605) : (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>) -> !tuple
    return %606 : !tuple
  }
}

