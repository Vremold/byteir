// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
!tuple = type tuple<tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>>
module  {
  func @main(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<32x3x224x224xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64x64x3x3xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64x64x3x3xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64xf32>, %arg15: tensor<64xf32>, %arg16: tensor<64x64x3x3xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64xf32>, %arg21: tensor<64x64x3x3xf32>, %arg22: tensor<64xf32>, %arg23: tensor<64xf32>, %arg24: tensor<64xf32>, %arg25: tensor<64xf32>, %arg26: tensor<128x64x3x3xf32>, %arg27: tensor<128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128x3x3xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<128xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128x64x1x1xf32>, %arg41: tensor<128x128x3x3xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<128xf32>, %arg46: tensor<128x128x3x3xf32>, %arg47: tensor<128xf32>, %arg48: tensor<128xf32>, %arg49: tensor<128xf32>, %arg50: tensor<128xf32>, %arg51: tensor<256x128x3x3xf32>, %arg52: tensor<256xf32>, %arg53: tensor<256xf32>, %arg54: tensor<256xf32>, %arg55: tensor<256xf32>, %arg56: tensor<256x256x3x3xf32>, %arg57: tensor<256xf32>, %arg58: tensor<256xf32>, %arg59: tensor<256xf32>, %arg60: tensor<256xf32>, %arg61: tensor<256xf32>, %arg62: tensor<256xf32>, %arg63: tensor<256xf32>, %arg64: tensor<256xf32>, %arg65: tensor<256x128x1x1xf32>, %arg66: tensor<256x256x3x3xf32>, %arg67: tensor<256xf32>, %arg68: tensor<256xf32>, %arg69: tensor<256xf32>, %arg70: tensor<256xf32>, %arg71: tensor<256x256x3x3xf32>, %arg72: tensor<256xf32>, %arg73: tensor<256xf32>, %arg74: tensor<256xf32>, %arg75: tensor<256xf32>, %arg76: tensor<512x256x3x3xf32>, %arg77: tensor<512xf32>, %arg78: tensor<512xf32>, %arg79: tensor<512xf32>, %arg80: tensor<512xf32>, %arg81: tensor<512x512x3x3xf32>, %arg82: tensor<512xf32>, %arg83: tensor<512xf32>, %arg84: tensor<512xf32>, %arg85: tensor<512xf32>, %arg86: tensor<512xf32>, %arg87: tensor<512xf32>, %arg88: tensor<512xf32>, %arg89: tensor<512xf32>, %arg90: tensor<512x256x1x1xf32>, %arg91: tensor<512x512x3x3xf32>, %arg92: tensor<512xf32>, %arg93: tensor<512xf32>, %arg94: tensor<512xf32>, %arg95: tensor<512xf32>, %arg96: tensor<512x512x3x3xf32>, %arg97: tensor<512xf32>, %arg98: tensor<512xf32>, %arg99: tensor<512xf32>, %arg100: tensor<512xf32>, %arg101: tensor<1000x512xf32>, %arg102: tensor<32x1000xf16>, %arg103: tensor<1000xf32>) -> !tuple {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
    %1 = "mhlo.convert"(%arg0) : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf16>
    %2 = call @aten.convolution_overrideable.9(%0, %1) : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16>
    %3 = call @aten.native_batch_norm.14(%2, %arg5, %arg4, %arg3, %arg2) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x112x112xf16>
    %5 = call @aten.relu.36(%4) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %6 = call @aten.max_pool2d.121(%5) : (tensor<32x64x112x112xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>>
    %7 = "mhlo.get_tuple_element"(%6) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>>) -> tensor<32x64x56x56xui32>
    %8 = "mhlo.convert"(%arg101) : (tensor<1000x512xf32>) -> tensor<1000x512xf16>
    %9 = call @aten.permute.618(%8) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %10 = call @aten.permute.622(%9) : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %11 = call @aten.mm.627(%arg102, %10) : (tensor<32x1000xf16>, tensor<1000x512xf16>) -> tensor<32x512xf16>
    %12 = call @aten.view.632(%11) : (tensor<32x512xf16>) -> tensor<32x512x1x1xf16>
    %13 = call @aten.expand.636(%12) : (tensor<32x512x1x1xf16>) -> tensor<32x512x7x7xf16>
    %14 = mhlo.constant dense<4.900000e+01> : tensor<f16>
    %15 = call @aten.div.642(%13, %14) : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512x7x7xf16>
    %16 = "mhlo.get_tuple_element"(%6) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>>) -> tensor<32x64x56x56xf16>
    %17 = "mhlo.convert"(%arg6) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %18 = call @aten.convolution_overrideable.153(%16, %17) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %19 = call @aten.native_batch_norm.158(%18, %arg10, %arg9, %arg8, %arg7) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %20 = "mhlo.get_tuple_element"(%19) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %21 = call @aten.relu.180(%20) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %22 = "mhlo.convert"(%arg11) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %23 = call @aten.convolution_overrideable.153(%21, %22) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %24 = call @aten.native_batch_norm.158(%23, %arg15, %arg14, %arg13, %arg12) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %25 = "mhlo.get_tuple_element"(%24) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %26 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %27 = call @aten.expand.43(%26) : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %28 = call @aten.mul.200(%16, %27) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %29 = call @aten.add.205(%25, %28) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %30 = call @aten.relu.180(%29) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %31 = "mhlo.convert"(%arg16) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %32 = call @aten.convolution_overrideable.153(%30, %31) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %33 = call @aten.native_batch_norm.158(%32, %arg20, %arg19, %arg18, %arg17) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %34 = "mhlo.get_tuple_element"(%33) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %35 = call @aten.relu.180(%34) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %36 = "mhlo.convert"(%arg21) : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf16>
    %37 = call @aten.convolution_overrideable.153(%35, %36) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    %38 = call @aten.native_batch_norm.158(%37, %arg25, %arg24, %arg23, %arg22) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    %39 = "mhlo.get_tuple_element"(%38) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %40 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %41 = call @aten.expand.43(%40) : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %42 = call @aten.mul.200(%30, %41) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %43 = call @aten.add.205(%39, %42) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %44 = call @aten.relu.180(%43) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %45 = "mhlo.convert"(%arg26) : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf16>
    %46 = call @aten.convolution_overrideable.251(%44, %45) : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16>
    %47 = call @aten.native_batch_norm.256(%46, %arg30, %arg29, %arg28, %arg27) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %48 = "mhlo.get_tuple_element"(%47) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %49 = call @aten.relu.278(%48) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %50 = "mhlo.convert"(%arg31) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %51 = call @aten.convolution_overrideable.290(%49, %50) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %52 = call @aten.native_batch_norm.256(%51, %arg35, %arg34, %arg33, %arg32) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %53 = "mhlo.get_tuple_element"(%52) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %54 = "mhlo.convert"(%arg40) : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf16>
    %55 = call @aten.convolution_overrideable.314(%44, %54) : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16>
    %56 = call @aten.native_batch_norm.256(%55, %arg39, %arg38, %arg37, %arg36) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %57 = "mhlo.get_tuple_element"(%56) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %58 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %59 = call @aten.expand.301(%58) : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %60 = call @aten.mul.324(%57, %59) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %61 = call @aten.add.329(%53, %60) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %62 = call @aten.relu.278(%61) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %63 = "mhlo.convert"(%arg41) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %64 = call @aten.convolution_overrideable.290(%62, %63) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %65 = call @aten.native_batch_norm.256(%64, %arg45, %arg44, %arg43, %arg42) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %66 = "mhlo.get_tuple_element"(%65) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %67 = call @aten.relu.278(%66) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %68 = "mhlo.convert"(%arg46) : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf16>
    %69 = call @aten.convolution_overrideable.290(%67, %68) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    %70 = call @aten.native_batch_norm.256(%69, %arg50, %arg49, %arg48, %arg47) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    %71 = "mhlo.get_tuple_element"(%70) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %72 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %73 = call @aten.expand.301(%72) : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %74 = call @aten.mul.324(%62, %73) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %75 = call @aten.add.329(%71, %74) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %76 = call @aten.relu.278(%75) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %77 = "mhlo.convert"(%arg51) : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf16>
    %78 = call @aten.convolution_overrideable.375(%76, %77) : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16>
    %79 = call @aten.native_batch_norm.380(%78, %arg55, %arg54, %arg53, %arg52) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %80 = "mhlo.get_tuple_element"(%79) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %81 = call @aten.relu.402(%80) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %82 = "mhlo.convert"(%arg56) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %83 = call @aten.convolution_overrideable.414(%81, %82) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %84 = call @aten.native_batch_norm.380(%83, %arg60, %arg59, %arg58, %arg57) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %85 = "mhlo.get_tuple_element"(%84) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %86 = "mhlo.convert"(%arg65) : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf16>
    %87 = call @aten.convolution_overrideable.438(%76, %86) : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16>
    %88 = call @aten.native_batch_norm.380(%87, %arg64, %arg63, %arg62, %arg61) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %89 = "mhlo.get_tuple_element"(%88) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %90 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %91 = call @aten.expand.425(%90) : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %92 = call @aten.mul.448(%89, %91) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %93 = call @aten.add.453(%85, %92) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %94 = call @aten.relu.402(%93) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %95 = "mhlo.convert"(%arg66) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %96 = call @aten.convolution_overrideable.414(%94, %95) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %97 = call @aten.native_batch_norm.380(%96, %arg70, %arg69, %arg68, %arg67) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %98 = "mhlo.get_tuple_element"(%97) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %99 = call @aten.relu.402(%98) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %100 = "mhlo.convert"(%arg71) : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf16>
    %101 = call @aten.convolution_overrideable.414(%99, %100) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    %102 = call @aten.native_batch_norm.380(%101, %arg75, %arg74, %arg73, %arg72) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    %103 = "mhlo.get_tuple_element"(%102) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %104 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %105 = call @aten.expand.425(%104) : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %106 = call @aten.mul.448(%94, %105) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %107 = call @aten.add.453(%103, %106) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %108 = call @aten.relu.402(%107) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %109 = "mhlo.convert"(%arg76) : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf16>
    %110 = call @aten.convolution_overrideable.499(%108, %109) : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16>
    %111 = call @aten.native_batch_norm.504(%110, %arg80, %arg79, %arg78, %arg77) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %112 = "mhlo.get_tuple_element"(%111) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %113 = call @aten.relu.526(%112) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %114 = "mhlo.convert"(%arg81) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %115 = call @aten.convolution_overrideable.538(%113, %114) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %116 = call @aten.native_batch_norm.504(%115, %arg85, %arg84, %arg83, %arg82) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %117 = "mhlo.get_tuple_element"(%116) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %118 = "mhlo.convert"(%arg90) : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf16>
    %119 = call @aten.convolution_overrideable.562(%108, %118) : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16>
    %120 = call @aten.native_batch_norm.504(%119, %arg89, %arg88, %arg87, %arg86) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %121 = "mhlo.get_tuple_element"(%120) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %122 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %123 = call @aten.expand.549(%122) : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %124 = call @aten.mul.572(%121, %123) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %125 = call @aten.add.577(%117, %124) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %126 = call @aten.relu.526(%125) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %127 = "mhlo.convert"(%arg91) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %128 = call @aten.convolution_overrideable.538(%126, %127) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %129 = call @aten.native_batch_norm.504(%128, %arg95, %arg94, %arg93, %arg92) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %130 = "mhlo.get_tuple_element"(%129) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %131 = call @aten.relu.526(%130) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %132 = "mhlo.convert"(%arg96) : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf16>
    %133 = call @aten.convolution_overrideable.538(%131, %132) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    %134 = call @aten.native_batch_norm.504(%133, %arg100, %arg99, %arg98, %arg97) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    %135 = "mhlo.get_tuple_element"(%134) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %136 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %137 = call @aten.expand.549(%136) : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %138 = call @aten.mul.572(%126, %137) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %139 = call @aten.add.577(%135, %138) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %140 = call @aten.relu.526(%139) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %141 = call @aten.threshold_backward.648(%15, %140) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %142 = "mhlo.get_tuple_element"(%134) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %143 = "mhlo.get_tuple_element"(%134) {index = 3 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %144 = call @aten.native_batch_norm_backward.658(%141, %133, %arg100, %142, %143) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %145 = "mhlo.get_tuple_element"(%144) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %146 = call @aten.convolution_backward_overrideable.687(%145, %131, %132) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %147 = "mhlo.get_tuple_element"(%146) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %148 = "mhlo.get_tuple_element"(%146) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<32x512x7x7xf16>
    %149 = call @aten.threshold_backward.648(%148, %131) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %150 = "mhlo.get_tuple_element"(%129) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %151 = "mhlo.get_tuple_element"(%129) {index = 3 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %152 = call @aten.native_batch_norm_backward.658(%149, %128, %arg95, %150, %151) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %153 = "mhlo.get_tuple_element"(%152) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %154 = call @aten.convolution_backward_overrideable.687(%153, %126, %127) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %155 = "mhlo.get_tuple_element"(%154) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %156 = "mhlo.get_tuple_element"(%154) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<32x512x7x7xf16>
    %157 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %158 = call @aten.expand.549(%157) : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %159 = call @aten.mul.572(%156, %158) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %160 = call @aten.add.577(%141, %159) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %161 = call @aten.threshold_backward.648(%160, %126) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %162 = "mhlo.get_tuple_element"(%116) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %163 = "mhlo.get_tuple_element"(%116) {index = 3 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %164 = call @aten.native_batch_norm_backward.658(%161, %115, %arg85, %162, %163) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %165 = "mhlo.get_tuple_element"(%164) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %166 = call @aten.convolution_backward_overrideable.687(%165, %113, %114) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    %167 = "mhlo.get_tuple_element"(%166) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %168 = "mhlo.get_tuple_element"(%166) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<32x512x7x7xf16>
    %169 = call @aten.threshold_backward.648(%168, %113) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    %170 = "mhlo.get_tuple_element"(%111) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %171 = "mhlo.get_tuple_element"(%111) {index = 3 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %172 = call @aten.native_batch_norm_backward.658(%169, %110, %arg80, %170, %171) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %173 = "mhlo.get_tuple_element"(%172) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %174 = call @aten.convolution_backward_overrideable.732(%173, %108, %109) : (tensor<32x512x7x7xf16>, tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    %175 = "mhlo.get_tuple_element"(%174) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %176 = "mhlo.get_tuple_element"(%120) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %177 = "mhlo.get_tuple_element"(%120) {index = 3 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %178 = call @aten.native_batch_norm_backward.658(%161, %119, %arg89, %176, %177) : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    %179 = "mhlo.get_tuple_element"(%178) {index = 0 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<32x512x7x7xf16>
    %180 = call @aten.convolution_backward_overrideable.757(%179, %108, %118) : (tensor<32x512x7x7xf16>, tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    %181 = "mhlo.get_tuple_element"(%180) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512xf16>
    %182 = "mhlo.get_tuple_element"(%180) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<32x256x14x14xf16>
    %183 = "mhlo.get_tuple_element"(%174) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<32x256x14x14xf16>
    %184 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %185 = call @aten.expand.425(%184) : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %186 = call @aten.mul.448(%183, %185) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %187 = call @aten.add.453(%182, %186) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %188 = call @aten.threshold_backward.774(%187, %108) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %189 = "mhlo.get_tuple_element"(%102) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %190 = "mhlo.get_tuple_element"(%102) {index = 3 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %191 = call @aten.native_batch_norm_backward.784(%188, %101, %arg75, %189, %190) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %192 = "mhlo.get_tuple_element"(%191) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %193 = call @aten.convolution_backward_overrideable.813(%192, %99, %100) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %194 = "mhlo.get_tuple_element"(%193) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %195 = "mhlo.get_tuple_element"(%193) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<32x256x14x14xf16>
    %196 = call @aten.threshold_backward.774(%195, %99) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %197 = "mhlo.get_tuple_element"(%97) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %198 = "mhlo.get_tuple_element"(%97) {index = 3 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %199 = call @aten.native_batch_norm_backward.784(%196, %96, %arg70, %197, %198) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %200 = "mhlo.get_tuple_element"(%199) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %201 = call @aten.convolution_backward_overrideable.813(%200, %94, %95) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %202 = "mhlo.get_tuple_element"(%201) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %203 = "mhlo.get_tuple_element"(%201) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<32x256x14x14xf16>
    %204 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %205 = call @aten.expand.425(%204) : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %206 = call @aten.mul.448(%203, %205) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %207 = call @aten.add.453(%188, %206) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %208 = call @aten.threshold_backward.774(%207, %94) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %209 = "mhlo.get_tuple_element"(%84) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %210 = "mhlo.get_tuple_element"(%84) {index = 3 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %211 = call @aten.native_batch_norm_backward.784(%208, %83, %arg60, %209, %210) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %212 = "mhlo.get_tuple_element"(%211) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %213 = call @aten.convolution_backward_overrideable.813(%212, %81, %82) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    %214 = "mhlo.get_tuple_element"(%213) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %215 = "mhlo.get_tuple_element"(%213) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<32x256x14x14xf16>
    %216 = call @aten.threshold_backward.774(%215, %81) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    %217 = "mhlo.get_tuple_element"(%79) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %218 = "mhlo.get_tuple_element"(%79) {index = 3 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %219 = call @aten.native_batch_norm_backward.784(%216, %78, %arg55, %217, %218) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %220 = "mhlo.get_tuple_element"(%219) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %221 = call @aten.convolution_backward_overrideable.858(%220, %76, %77) : (tensor<32x256x14x14xf16>, tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    %222 = "mhlo.get_tuple_element"(%221) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %223 = "mhlo.get_tuple_element"(%88) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %224 = "mhlo.get_tuple_element"(%88) {index = 3 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %225 = call @aten.native_batch_norm_backward.784(%208, %87, %arg64, %223, %224) : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    %226 = "mhlo.get_tuple_element"(%225) {index = 0 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<32x256x14x14xf16>
    %227 = call @aten.convolution_backward_overrideable.883(%226, %76, %86) : (tensor<32x256x14x14xf16>, tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    %228 = "mhlo.get_tuple_element"(%227) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256xf16>
    %229 = "mhlo.get_tuple_element"(%227) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<32x128x28x28xf16>
    %230 = "mhlo.get_tuple_element"(%221) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<32x128x28x28xf16>
    %231 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %232 = call @aten.expand.301(%231) : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %233 = call @aten.mul.324(%230, %232) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %234 = call @aten.add.329(%229, %233) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %235 = call @aten.threshold_backward.900(%234, %76) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %236 = "mhlo.get_tuple_element"(%70) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %237 = "mhlo.get_tuple_element"(%70) {index = 3 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %238 = call @aten.native_batch_norm_backward.910(%235, %69, %arg50, %236, %237) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %239 = "mhlo.get_tuple_element"(%238) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %240 = call @aten.convolution_backward_overrideable.939(%239, %67, %68) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %241 = "mhlo.get_tuple_element"(%240) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %242 = "mhlo.get_tuple_element"(%240) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<32x128x28x28xf16>
    %243 = call @aten.threshold_backward.900(%242, %67) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %244 = "mhlo.get_tuple_element"(%65) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %245 = "mhlo.get_tuple_element"(%65) {index = 3 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %246 = call @aten.native_batch_norm_backward.910(%243, %64, %arg45, %244, %245) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %247 = "mhlo.get_tuple_element"(%246) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %248 = call @aten.convolution_backward_overrideable.939(%247, %62, %63) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %249 = "mhlo.get_tuple_element"(%248) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %250 = "mhlo.get_tuple_element"(%248) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<32x128x28x28xf16>
    %251 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %252 = call @aten.expand.301(%251) : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %253 = call @aten.mul.324(%250, %252) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %254 = call @aten.add.329(%235, %253) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %255 = call @aten.threshold_backward.900(%254, %62) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %256 = "mhlo.get_tuple_element"(%52) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %257 = "mhlo.get_tuple_element"(%52) {index = 3 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %258 = call @aten.native_batch_norm_backward.910(%255, %51, %arg35, %256, %257) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %259 = "mhlo.get_tuple_element"(%258) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %260 = call @aten.convolution_backward_overrideable.939(%259, %49, %50) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    %261 = "mhlo.get_tuple_element"(%260) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %262 = "mhlo.get_tuple_element"(%260) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<32x128x28x28xf16>
    %263 = call @aten.threshold_backward.900(%262, %49) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    %264 = "mhlo.get_tuple_element"(%47) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %265 = "mhlo.get_tuple_element"(%47) {index = 3 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %266 = call @aten.native_batch_norm_backward.910(%263, %46, %arg30, %264, %265) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %267 = "mhlo.get_tuple_element"(%266) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %268 = call @aten.convolution_backward_overrideable.984(%267, %44, %45) : (tensor<32x128x28x28xf16>, tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    %269 = "mhlo.get_tuple_element"(%268) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %270 = "mhlo.get_tuple_element"(%56) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %271 = "mhlo.get_tuple_element"(%56) {index = 3 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %272 = call @aten.native_batch_norm_backward.910(%255, %55, %arg39, %270, %271) : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    %273 = "mhlo.get_tuple_element"(%272) {index = 0 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<32x128x28x28xf16>
    %274 = call @aten.convolution_backward_overrideable.1009(%273, %44, %54) : (tensor<32x128x28x28xf16>, tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    %275 = "mhlo.get_tuple_element"(%274) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128xf16>
    %276 = "mhlo.get_tuple_element"(%274) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<32x64x56x56xf16>
    %277 = "mhlo.get_tuple_element"(%268) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<32x64x56x56xf16>
    %278 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %279 = call @aten.expand.43(%278) : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %280 = call @aten.mul.200(%277, %279) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %281 = call @aten.add.205(%276, %280) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %282 = call @aten.threshold_backward.1026(%281, %44) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %283 = "mhlo.get_tuple_element"(%38) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %284 = "mhlo.get_tuple_element"(%38) {index = 3 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %285 = call @aten.native_batch_norm_backward.1036(%282, %37, %arg25, %283, %284) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %286 = "mhlo.get_tuple_element"(%285) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %287 = call @aten.convolution_backward_overrideable.1065(%286, %35, %36) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %288 = "mhlo.get_tuple_element"(%287) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %289 = "mhlo.get_tuple_element"(%287) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<32x64x56x56xf16>
    %290 = call @aten.threshold_backward.1026(%289, %35) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %291 = "mhlo.get_tuple_element"(%33) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %292 = "mhlo.get_tuple_element"(%33) {index = 3 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %293 = call @aten.native_batch_norm_backward.1036(%290, %32, %arg20, %291, %292) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %294 = "mhlo.get_tuple_element"(%293) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %295 = call @aten.convolution_backward_overrideable.1065(%294, %30, %31) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %296 = "mhlo.get_tuple_element"(%295) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %297 = "mhlo.get_tuple_element"(%295) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<32x64x56x56xf16>
    %298 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %299 = call @aten.expand.43(%298) : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %300 = call @aten.mul.200(%297, %299) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %301 = call @aten.add.205(%282, %300) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %302 = call @aten.threshold_backward.1026(%301, %30) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %303 = "mhlo.get_tuple_element"(%24) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %304 = "mhlo.get_tuple_element"(%24) {index = 3 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %305 = call @aten.native_batch_norm_backward.1036(%302, %23, %arg15, %303, %304) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %306 = "mhlo.get_tuple_element"(%305) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %307 = call @aten.convolution_backward_overrideable.1065(%306, %21, %22) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %308 = "mhlo.get_tuple_element"(%307) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %309 = "mhlo.get_tuple_element"(%307) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<32x64x56x56xf16>
    %310 = call @aten.threshold_backward.1026(%309, %21) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %311 = "mhlo.get_tuple_element"(%19) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %312 = "mhlo.get_tuple_element"(%19) {index = 3 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %313 = call @aten.native_batch_norm_backward.1036(%310, %18, %arg10, %311, %312) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    %314 = "mhlo.get_tuple_element"(%313) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x56x56xf16>
    %315 = call @aten.convolution_backward_overrideable.1065(%314, %16, %17) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    %316 = "mhlo.get_tuple_element"(%315) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %317 = "mhlo.get_tuple_element"(%315) {index = 0 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<32x64x56x56xf16>
    %318 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %319 = call @aten.expand.43(%318) : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %320 = call @aten.mul.200(%317, %319) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %321 = call @aten.add.205(%302, %320) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    %322 = call @aten.max_pool2d_with_indices_backward.1120(%321, %5) : (tensor<32x64x56x56xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %323 = call @aten.threshold_backward.1126(%322, %5) : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    %324 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %325 = "mhlo.get_tuple_element"(%3) {index = 3 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %326 = call @aten.native_batch_norm_backward.1136(%323, %2, %arg5, %324, %325) : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    %327 = "mhlo.get_tuple_element"(%326) {index = 0 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<32x64x112x112xf16>
    %328 = call @aten.convolution_backward_overrideable.1165(%327, %0, %1) : (tensor<32x64x112x112xf16>, tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    %329 = "mhlo.get_tuple_element"(%328) {index = 0 : i32} : (tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<32x3x224x224xf16>
    %330 = "mhlo.get_tuple_element"(%328) {index = 2 : i32} : (tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64xf16>
    %331 = "mhlo.get_tuple_element"(%328) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>) -> tensor<64x3x7x7xf16>
    %332 = "mhlo.convert"(%331) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x3x7x7xf16>) -> tensor<64x3x7x7xf32>
    %333 = "mhlo.get_tuple_element"(%326) {index = 1 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %334 = "mhlo.get_tuple_element"(%326) {index = 2 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %335 = "mhlo.get_tuple_element"(%315) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %336 = "mhlo.convert"(%335) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %337 = "mhlo.get_tuple_element"(%313) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %338 = "mhlo.get_tuple_element"(%313) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %339 = "mhlo.get_tuple_element"(%307) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %340 = "mhlo.convert"(%339) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %341 = "mhlo.get_tuple_element"(%305) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %342 = "mhlo.get_tuple_element"(%305) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %343 = "mhlo.get_tuple_element"(%295) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %344 = "mhlo.convert"(%343) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %345 = "mhlo.get_tuple_element"(%293) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %346 = "mhlo.get_tuple_element"(%293) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %347 = "mhlo.get_tuple_element"(%287) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>) -> tensor<64x64x3x3xf16>
    %348 = "mhlo.convert"(%347) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<64x64x3x3xf16>) -> tensor<64x64x3x3xf32>
    %349 = "mhlo.get_tuple_element"(%285) {index = 1 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %350 = "mhlo.get_tuple_element"(%285) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %351 = "mhlo.get_tuple_element"(%268) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>) -> tensor<128x64x3x3xf16>
    %352 = "mhlo.convert"(%351) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x3x3xf16>) -> tensor<128x64x3x3xf32>
    %353 = "mhlo.get_tuple_element"(%266) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %354 = "mhlo.get_tuple_element"(%266) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %355 = "mhlo.get_tuple_element"(%260) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %356 = "mhlo.convert"(%355) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %357 = "mhlo.get_tuple_element"(%258) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %358 = "mhlo.get_tuple_element"(%258) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %359 = "mhlo.get_tuple_element"(%274) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>) -> tensor<128x64x1x1xf16>
    %360 = "mhlo.convert"(%359) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x64x1x1xf16>) -> tensor<128x64x1x1xf32>
    %361 = "mhlo.get_tuple_element"(%272) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %362 = "mhlo.get_tuple_element"(%272) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %363 = "mhlo.get_tuple_element"(%248) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %364 = "mhlo.convert"(%363) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %365 = "mhlo.get_tuple_element"(%246) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %366 = "mhlo.get_tuple_element"(%246) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %367 = "mhlo.get_tuple_element"(%240) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>) -> tensor<128x128x3x3xf16>
    %368 = "mhlo.convert"(%367) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<128x128x3x3xf16>) -> tensor<128x128x3x3xf32>
    %369 = "mhlo.get_tuple_element"(%238) {index = 1 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %370 = "mhlo.get_tuple_element"(%238) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %371 = "mhlo.get_tuple_element"(%221) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>) -> tensor<256x128x3x3xf16>
    %372 = "mhlo.convert"(%371) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x3x3xf16>) -> tensor<256x128x3x3xf32>
    %373 = "mhlo.get_tuple_element"(%219) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %374 = "mhlo.get_tuple_element"(%219) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %375 = "mhlo.get_tuple_element"(%213) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %376 = "mhlo.convert"(%375) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %377 = "mhlo.get_tuple_element"(%211) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %378 = "mhlo.get_tuple_element"(%211) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %379 = "mhlo.get_tuple_element"(%227) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>) -> tensor<256x128x1x1xf16>
    %380 = "mhlo.convert"(%379) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x128x1x1xf16>) -> tensor<256x128x1x1xf32>
    %381 = "mhlo.get_tuple_element"(%225) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %382 = "mhlo.get_tuple_element"(%225) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %383 = "mhlo.get_tuple_element"(%201) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %384 = "mhlo.convert"(%383) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %385 = "mhlo.get_tuple_element"(%199) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %386 = "mhlo.get_tuple_element"(%199) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %387 = "mhlo.get_tuple_element"(%193) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>) -> tensor<256x256x3x3xf16>
    %388 = "mhlo.convert"(%387) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<256x256x3x3xf16>) -> tensor<256x256x3x3xf32>
    %389 = "mhlo.get_tuple_element"(%191) {index = 1 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %390 = "mhlo.get_tuple_element"(%191) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %391 = "mhlo.get_tuple_element"(%174) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>) -> tensor<512x256x3x3xf16>
    %392 = "mhlo.convert"(%391) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x3x3xf16>) -> tensor<512x256x3x3xf32>
    %393 = "mhlo.get_tuple_element"(%172) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %394 = "mhlo.get_tuple_element"(%172) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %395 = "mhlo.get_tuple_element"(%166) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %396 = "mhlo.convert"(%395) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %397 = "mhlo.get_tuple_element"(%164) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %398 = "mhlo.get_tuple_element"(%164) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %399 = "mhlo.get_tuple_element"(%180) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>) -> tensor<512x256x1x1xf16>
    %400 = "mhlo.convert"(%399) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x256x1x1xf16>) -> tensor<512x256x1x1xf32>
    %401 = "mhlo.get_tuple_element"(%178) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %402 = "mhlo.get_tuple_element"(%178) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %403 = "mhlo.get_tuple_element"(%154) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %404 = "mhlo.convert"(%403) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %405 = "mhlo.get_tuple_element"(%152) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %406 = "mhlo.get_tuple_element"(%152) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %407 = "mhlo.get_tuple_element"(%146) {index = 1 : i32, minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>) -> tensor<512x512x3x3xf16>
    %408 = "mhlo.convert"(%407) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>} : (tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf32>
    %409 = "mhlo.get_tuple_element"(%144) {index = 1 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %410 = "mhlo.get_tuple_element"(%144) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %411 = call @aten.mean.1205(%140) : (tensor<32x512x7x7xf16>) -> tensor<32x512x1x1xf16>
    %412 = call @aten.view.1222(%411) : (tensor<32x512x1x1xf16>) -> tensor<32x512xf16>
    %413 = call @aten.permute.1226(%412) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<32x512xf16>) -> tensor<512x32xf16>
    %414 = call @aten.mm.1230(%413, %arg102) : (tensor<512x32xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    %415 = call @aten.permute.1235(%414) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    %416 = "mhlo.convert"(%415) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<1000x512xf32>
    %417 = call @aten.sum.1244(%arg102) : (tensor<32x1000xf16>) -> tensor<1x1000xf32>
    %418 = call @aten.view.1252(%417) : (tensor<1x1000xf32>) -> tensor<1000xf32>
    %419 = "mhlo.convert"(%418) : (tensor<1000xf32>) -> tensor<1000xf16>
    %420 = "mhlo.convert"(%419) : (tensor<1000xf16>) -> tensor<1000xf32>
    %421 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %422 = "mhlo.broadcast_in_dim"(%421) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %423 = mhlo.multiply %324, %422 : tensor<64xf32>
    %424 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %425 = mhlo.subtract %424, %421 : tensor<f32>
    %426 = "mhlo.broadcast_in_dim"(%425) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %427 = mhlo.multiply %arg3, %426 : tensor<64xf32>
    %428 = mhlo.add %423, %427 : tensor<64xf32>
    %429 = "mhlo.get_tuple_element"(%3) {index = 2 : i32} : (tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %430 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %431 = "mhlo.broadcast_in_dim"(%430) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %432 = mhlo.multiply %429, %431 : tensor<64xf32>
    %433 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %434 = mhlo.subtract %433, %430 : tensor<f32>
    %435 = "mhlo.broadcast_in_dim"(%434) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %436 = mhlo.multiply %arg2, %435 : tensor<64xf32>
    %437 = mhlo.add %432, %436 : tensor<64xf32>
    %438 = mhlo.constant dense<0> : tensor<i64>
    %439 = mhlo.constant dense<1> : tensor<i64>
    %440 = mhlo.constant dense<1> : tensor<i64>
    %441 = call @aten.mul.1276(%439, %440) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %442 = call @aten.add.1282(%438, %441) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %443 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %444 = "mhlo.broadcast_in_dim"(%443) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %445 = mhlo.multiply %311, %444 : tensor<64xf32>
    %446 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %447 = mhlo.subtract %446, %443 : tensor<f32>
    %448 = "mhlo.broadcast_in_dim"(%447) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %449 = mhlo.multiply %arg8, %448 : tensor<64xf32>
    %450 = mhlo.add %445, %449 : tensor<64xf32>
    %451 = "mhlo.get_tuple_element"(%19) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %452 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %453 = "mhlo.broadcast_in_dim"(%452) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %454 = mhlo.multiply %451, %453 : tensor<64xf32>
    %455 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %456 = mhlo.subtract %455, %452 : tensor<f32>
    %457 = "mhlo.broadcast_in_dim"(%456) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %458 = mhlo.multiply %arg7, %457 : tensor<64xf32>
    %459 = mhlo.add %454, %458 : tensor<64xf32>
    %460 = mhlo.constant dense<0> : tensor<i64>
    %461 = mhlo.constant dense<1> : tensor<i64>
    %462 = mhlo.constant dense<1> : tensor<i64>
    %463 = call @aten.mul.1276(%461, %462) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %464 = call @aten.add.1282(%460, %463) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %465 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %466 = "mhlo.broadcast_in_dim"(%465) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %467 = mhlo.multiply %303, %466 : tensor<64xf32>
    %468 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %469 = mhlo.subtract %468, %465 : tensor<f32>
    %470 = "mhlo.broadcast_in_dim"(%469) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %471 = mhlo.multiply %arg13, %470 : tensor<64xf32>
    %472 = mhlo.add %467, %471 : tensor<64xf32>
    %473 = "mhlo.get_tuple_element"(%24) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %474 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %475 = "mhlo.broadcast_in_dim"(%474) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %476 = mhlo.multiply %473, %475 : tensor<64xf32>
    %477 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %478 = mhlo.subtract %477, %474 : tensor<f32>
    %479 = "mhlo.broadcast_in_dim"(%478) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %480 = mhlo.multiply %arg12, %479 : tensor<64xf32>
    %481 = mhlo.add %476, %480 : tensor<64xf32>
    %482 = mhlo.constant dense<0> : tensor<i64>
    %483 = mhlo.constant dense<1> : tensor<i64>
    %484 = mhlo.constant dense<1> : tensor<i64>
    %485 = call @aten.mul.1276(%483, %484) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %486 = call @aten.add.1282(%482, %485) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %487 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %488 = "mhlo.broadcast_in_dim"(%487) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %489 = mhlo.multiply %291, %488 : tensor<64xf32>
    %490 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %491 = mhlo.subtract %490, %487 : tensor<f32>
    %492 = "mhlo.broadcast_in_dim"(%491) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %493 = mhlo.multiply %arg18, %492 : tensor<64xf32>
    %494 = mhlo.add %489, %493 : tensor<64xf32>
    %495 = "mhlo.get_tuple_element"(%33) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %496 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %497 = "mhlo.broadcast_in_dim"(%496) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %498 = mhlo.multiply %495, %497 : tensor<64xf32>
    %499 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %500 = mhlo.subtract %499, %496 : tensor<f32>
    %501 = "mhlo.broadcast_in_dim"(%500) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %502 = mhlo.multiply %arg17, %501 : tensor<64xf32>
    %503 = mhlo.add %498, %502 : tensor<64xf32>
    %504 = mhlo.constant dense<0> : tensor<i64>
    %505 = mhlo.constant dense<1> : tensor<i64>
    %506 = mhlo.constant dense<1> : tensor<i64>
    %507 = call @aten.mul.1276(%505, %506) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %508 = call @aten.add.1282(%504, %507) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %509 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %510 = "mhlo.broadcast_in_dim"(%509) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %511 = mhlo.multiply %283, %510 : tensor<64xf32>
    %512 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %513 = mhlo.subtract %512, %509 : tensor<f32>
    %514 = "mhlo.broadcast_in_dim"(%513) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %515 = mhlo.multiply %arg23, %514 : tensor<64xf32>
    %516 = mhlo.add %511, %515 : tensor<64xf32>
    %517 = "mhlo.get_tuple_element"(%38) {index = 2 : i32} : (tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>) -> tensor<64xf32>
    %518 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %519 = "mhlo.broadcast_in_dim"(%518) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %520 = mhlo.multiply %517, %519 : tensor<64xf32>
    %521 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %522 = mhlo.subtract %521, %518 : tensor<f32>
    %523 = "mhlo.broadcast_in_dim"(%522) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %524 = mhlo.multiply %arg22, %523 : tensor<64xf32>
    %525 = mhlo.add %520, %524 : tensor<64xf32>
    %526 = mhlo.constant dense<0> : tensor<i64>
    %527 = mhlo.constant dense<1> : tensor<i64>
    %528 = mhlo.constant dense<1> : tensor<i64>
    %529 = call @aten.mul.1276(%527, %528) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %530 = call @aten.add.1282(%526, %529) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %531 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %532 = "mhlo.broadcast_in_dim"(%531) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %533 = mhlo.multiply %264, %532 : tensor<128xf32>
    %534 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %535 = mhlo.subtract %534, %531 : tensor<f32>
    %536 = "mhlo.broadcast_in_dim"(%535) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %537 = mhlo.multiply %arg28, %536 : tensor<128xf32>
    %538 = mhlo.add %533, %537 : tensor<128xf32>
    %539 = "mhlo.get_tuple_element"(%47) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %540 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %541 = "mhlo.broadcast_in_dim"(%540) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %542 = mhlo.multiply %539, %541 : tensor<128xf32>
    %543 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %544 = mhlo.subtract %543, %540 : tensor<f32>
    %545 = "mhlo.broadcast_in_dim"(%544) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %546 = mhlo.multiply %arg27, %545 : tensor<128xf32>
    %547 = mhlo.add %542, %546 : tensor<128xf32>
    %548 = mhlo.constant dense<0> : tensor<i64>
    %549 = mhlo.constant dense<1> : tensor<i64>
    %550 = mhlo.constant dense<1> : tensor<i64>
    %551 = call @aten.mul.1276(%549, %550) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %552 = call @aten.add.1282(%548, %551) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %553 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %554 = "mhlo.broadcast_in_dim"(%553) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %555 = mhlo.multiply %256, %554 : tensor<128xf32>
    %556 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %557 = mhlo.subtract %556, %553 : tensor<f32>
    %558 = "mhlo.broadcast_in_dim"(%557) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %559 = mhlo.multiply %arg33, %558 : tensor<128xf32>
    %560 = mhlo.add %555, %559 : tensor<128xf32>
    %561 = "mhlo.get_tuple_element"(%52) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %562 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %563 = "mhlo.broadcast_in_dim"(%562) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %564 = mhlo.multiply %561, %563 : tensor<128xf32>
    %565 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %566 = mhlo.subtract %565, %562 : tensor<f32>
    %567 = "mhlo.broadcast_in_dim"(%566) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %568 = mhlo.multiply %arg32, %567 : tensor<128xf32>
    %569 = mhlo.add %564, %568 : tensor<128xf32>
    %570 = mhlo.constant dense<0> : tensor<i64>
    %571 = mhlo.constant dense<1> : tensor<i64>
    %572 = mhlo.constant dense<1> : tensor<i64>
    %573 = call @aten.mul.1276(%571, %572) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %574 = call @aten.add.1282(%570, %573) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %575 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %576 = "mhlo.broadcast_in_dim"(%575) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %577 = mhlo.multiply %270, %576 : tensor<128xf32>
    %578 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %579 = mhlo.subtract %578, %575 : tensor<f32>
    %580 = "mhlo.broadcast_in_dim"(%579) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %581 = mhlo.multiply %arg37, %580 : tensor<128xf32>
    %582 = mhlo.add %577, %581 : tensor<128xf32>
    %583 = "mhlo.get_tuple_element"(%56) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %584 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %585 = "mhlo.broadcast_in_dim"(%584) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %586 = mhlo.multiply %583, %585 : tensor<128xf32>
    %587 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %588 = mhlo.subtract %587, %584 : tensor<f32>
    %589 = "mhlo.broadcast_in_dim"(%588) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %590 = mhlo.multiply %arg36, %589 : tensor<128xf32>
    %591 = mhlo.add %586, %590 : tensor<128xf32>
    %592 = mhlo.constant dense<0> : tensor<i64>
    %593 = mhlo.constant dense<1> : tensor<i64>
    %594 = mhlo.constant dense<1> : tensor<i64>
    %595 = call @aten.mul.1276(%593, %594) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %596 = call @aten.add.1282(%592, %595) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %597 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %598 = "mhlo.broadcast_in_dim"(%597) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %599 = mhlo.multiply %244, %598 : tensor<128xf32>
    %600 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %601 = mhlo.subtract %600, %597 : tensor<f32>
    %602 = "mhlo.broadcast_in_dim"(%601) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %603 = mhlo.multiply %arg43, %602 : tensor<128xf32>
    %604 = mhlo.add %599, %603 : tensor<128xf32>
    %605 = "mhlo.get_tuple_element"(%65) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %606 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %607 = "mhlo.broadcast_in_dim"(%606) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %608 = mhlo.multiply %605, %607 : tensor<128xf32>
    %609 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %610 = mhlo.subtract %609, %606 : tensor<f32>
    %611 = "mhlo.broadcast_in_dim"(%610) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %612 = mhlo.multiply %arg42, %611 : tensor<128xf32>
    %613 = mhlo.add %608, %612 : tensor<128xf32>
    %614 = mhlo.constant dense<0> : tensor<i64>
    %615 = mhlo.constant dense<1> : tensor<i64>
    %616 = mhlo.constant dense<1> : tensor<i64>
    %617 = call @aten.mul.1276(%615, %616) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %618 = call @aten.add.1282(%614, %617) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %619 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %620 = "mhlo.broadcast_in_dim"(%619) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %621 = mhlo.multiply %236, %620 : tensor<128xf32>
    %622 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %623 = mhlo.subtract %622, %619 : tensor<f32>
    %624 = "mhlo.broadcast_in_dim"(%623) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %625 = mhlo.multiply %arg48, %624 : tensor<128xf32>
    %626 = mhlo.add %621, %625 : tensor<128xf32>
    %627 = "mhlo.get_tuple_element"(%70) {index = 2 : i32} : (tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>) -> tensor<128xf32>
    %628 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %629 = "mhlo.broadcast_in_dim"(%628) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %630 = mhlo.multiply %627, %629 : tensor<128xf32>
    %631 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %632 = mhlo.subtract %631, %628 : tensor<f32>
    %633 = "mhlo.broadcast_in_dim"(%632) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %634 = mhlo.multiply %arg47, %633 : tensor<128xf32>
    %635 = mhlo.add %630, %634 : tensor<128xf32>
    %636 = mhlo.constant dense<0> : tensor<i64>
    %637 = mhlo.constant dense<1> : tensor<i64>
    %638 = mhlo.constant dense<1> : tensor<i64>
    %639 = call @aten.mul.1276(%637, %638) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %640 = call @aten.add.1282(%636, %639) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %641 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %642 = "mhlo.broadcast_in_dim"(%641) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %643 = mhlo.multiply %217, %642 : tensor<256xf32>
    %644 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %645 = mhlo.subtract %644, %641 : tensor<f32>
    %646 = "mhlo.broadcast_in_dim"(%645) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %647 = mhlo.multiply %arg53, %646 : tensor<256xf32>
    %648 = mhlo.add %643, %647 : tensor<256xf32>
    %649 = "mhlo.get_tuple_element"(%79) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %650 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %651 = "mhlo.broadcast_in_dim"(%650) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %652 = mhlo.multiply %649, %651 : tensor<256xf32>
    %653 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %654 = mhlo.subtract %653, %650 : tensor<f32>
    %655 = "mhlo.broadcast_in_dim"(%654) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %656 = mhlo.multiply %arg52, %655 : tensor<256xf32>
    %657 = mhlo.add %652, %656 : tensor<256xf32>
    %658 = mhlo.constant dense<0> : tensor<i64>
    %659 = mhlo.constant dense<1> : tensor<i64>
    %660 = mhlo.constant dense<1> : tensor<i64>
    %661 = call @aten.mul.1276(%659, %660) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %662 = call @aten.add.1282(%658, %661) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %663 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %664 = "mhlo.broadcast_in_dim"(%663) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %665 = mhlo.multiply %209, %664 : tensor<256xf32>
    %666 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %667 = mhlo.subtract %666, %663 : tensor<f32>
    %668 = "mhlo.broadcast_in_dim"(%667) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %669 = mhlo.multiply %arg58, %668 : tensor<256xf32>
    %670 = mhlo.add %665, %669 : tensor<256xf32>
    %671 = "mhlo.get_tuple_element"(%84) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %672 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %673 = "mhlo.broadcast_in_dim"(%672) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %674 = mhlo.multiply %671, %673 : tensor<256xf32>
    %675 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %676 = mhlo.subtract %675, %672 : tensor<f32>
    %677 = "mhlo.broadcast_in_dim"(%676) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %678 = mhlo.multiply %arg57, %677 : tensor<256xf32>
    %679 = mhlo.add %674, %678 : tensor<256xf32>
    %680 = mhlo.constant dense<0> : tensor<i64>
    %681 = mhlo.constant dense<1> : tensor<i64>
    %682 = mhlo.constant dense<1> : tensor<i64>
    %683 = call @aten.mul.1276(%681, %682) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %684 = call @aten.add.1282(%680, %683) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %685 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %686 = "mhlo.broadcast_in_dim"(%685) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %687 = mhlo.multiply %223, %686 : tensor<256xf32>
    %688 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %689 = mhlo.subtract %688, %685 : tensor<f32>
    %690 = "mhlo.broadcast_in_dim"(%689) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %691 = mhlo.multiply %arg62, %690 : tensor<256xf32>
    %692 = mhlo.add %687, %691 : tensor<256xf32>
    %693 = "mhlo.get_tuple_element"(%88) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %694 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %695 = "mhlo.broadcast_in_dim"(%694) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %696 = mhlo.multiply %693, %695 : tensor<256xf32>
    %697 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %698 = mhlo.subtract %697, %694 : tensor<f32>
    %699 = "mhlo.broadcast_in_dim"(%698) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %700 = mhlo.multiply %arg61, %699 : tensor<256xf32>
    %701 = mhlo.add %696, %700 : tensor<256xf32>
    %702 = mhlo.constant dense<0> : tensor<i64>
    %703 = mhlo.constant dense<1> : tensor<i64>
    %704 = mhlo.constant dense<1> : tensor<i64>
    %705 = call @aten.mul.1276(%703, %704) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %706 = call @aten.add.1282(%702, %705) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %707 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %708 = "mhlo.broadcast_in_dim"(%707) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %709 = mhlo.multiply %197, %708 : tensor<256xf32>
    %710 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %711 = mhlo.subtract %710, %707 : tensor<f32>
    %712 = "mhlo.broadcast_in_dim"(%711) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %713 = mhlo.multiply %arg68, %712 : tensor<256xf32>
    %714 = mhlo.add %709, %713 : tensor<256xf32>
    %715 = "mhlo.get_tuple_element"(%97) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %716 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %717 = "mhlo.broadcast_in_dim"(%716) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %718 = mhlo.multiply %715, %717 : tensor<256xf32>
    %719 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %720 = mhlo.subtract %719, %716 : tensor<f32>
    %721 = "mhlo.broadcast_in_dim"(%720) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %722 = mhlo.multiply %arg67, %721 : tensor<256xf32>
    %723 = mhlo.add %718, %722 : tensor<256xf32>
    %724 = mhlo.constant dense<0> : tensor<i64>
    %725 = mhlo.constant dense<1> : tensor<i64>
    %726 = mhlo.constant dense<1> : tensor<i64>
    %727 = call @aten.mul.1276(%725, %726) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %728 = call @aten.add.1282(%724, %727) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %729 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %730 = "mhlo.broadcast_in_dim"(%729) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %731 = mhlo.multiply %189, %730 : tensor<256xf32>
    %732 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %733 = mhlo.subtract %732, %729 : tensor<f32>
    %734 = "mhlo.broadcast_in_dim"(%733) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %735 = mhlo.multiply %arg73, %734 : tensor<256xf32>
    %736 = mhlo.add %731, %735 : tensor<256xf32>
    %737 = "mhlo.get_tuple_element"(%102) {index = 2 : i32} : (tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>) -> tensor<256xf32>
    %738 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %739 = "mhlo.broadcast_in_dim"(%738) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %740 = mhlo.multiply %737, %739 : tensor<256xf32>
    %741 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %742 = mhlo.subtract %741, %738 : tensor<f32>
    %743 = "mhlo.broadcast_in_dim"(%742) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %744 = mhlo.multiply %arg72, %743 : tensor<256xf32>
    %745 = mhlo.add %740, %744 : tensor<256xf32>
    %746 = mhlo.constant dense<0> : tensor<i64>
    %747 = mhlo.constant dense<1> : tensor<i64>
    %748 = mhlo.constant dense<1> : tensor<i64>
    %749 = call @aten.mul.1276(%747, %748) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %750 = call @aten.add.1282(%746, %749) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %751 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %752 = "mhlo.broadcast_in_dim"(%751) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %753 = mhlo.multiply %170, %752 : tensor<512xf32>
    %754 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %755 = mhlo.subtract %754, %751 : tensor<f32>
    %756 = "mhlo.broadcast_in_dim"(%755) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %757 = mhlo.multiply %arg78, %756 : tensor<512xf32>
    %758 = mhlo.add %753, %757 : tensor<512xf32>
    %759 = "mhlo.get_tuple_element"(%111) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %760 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %761 = "mhlo.broadcast_in_dim"(%760) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %762 = mhlo.multiply %759, %761 : tensor<512xf32>
    %763 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %764 = mhlo.subtract %763, %760 : tensor<f32>
    %765 = "mhlo.broadcast_in_dim"(%764) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %766 = mhlo.multiply %arg77, %765 : tensor<512xf32>
    %767 = mhlo.add %762, %766 : tensor<512xf32>
    %768 = mhlo.constant dense<0> : tensor<i64>
    %769 = mhlo.constant dense<1> : tensor<i64>
    %770 = mhlo.constant dense<1> : tensor<i64>
    %771 = call @aten.mul.1276(%769, %770) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %772 = call @aten.add.1282(%768, %771) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %773 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %774 = "mhlo.broadcast_in_dim"(%773) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %775 = mhlo.multiply %162, %774 : tensor<512xf32>
    %776 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %777 = mhlo.subtract %776, %773 : tensor<f32>
    %778 = "mhlo.broadcast_in_dim"(%777) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %779 = mhlo.multiply %arg83, %778 : tensor<512xf32>
    %780 = mhlo.add %775, %779 : tensor<512xf32>
    %781 = "mhlo.get_tuple_element"(%116) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %782 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %783 = "mhlo.broadcast_in_dim"(%782) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %784 = mhlo.multiply %781, %783 : tensor<512xf32>
    %785 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %786 = mhlo.subtract %785, %782 : tensor<f32>
    %787 = "mhlo.broadcast_in_dim"(%786) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %788 = mhlo.multiply %arg82, %787 : tensor<512xf32>
    %789 = mhlo.add %784, %788 : tensor<512xf32>
    %790 = mhlo.constant dense<0> : tensor<i64>
    %791 = mhlo.constant dense<1> : tensor<i64>
    %792 = mhlo.constant dense<1> : tensor<i64>
    %793 = call @aten.mul.1276(%791, %792) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %794 = call @aten.add.1282(%790, %793) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %795 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %796 = "mhlo.broadcast_in_dim"(%795) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %797 = mhlo.multiply %176, %796 : tensor<512xf32>
    %798 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %799 = mhlo.subtract %798, %795 : tensor<f32>
    %800 = "mhlo.broadcast_in_dim"(%799) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %801 = mhlo.multiply %arg87, %800 : tensor<512xf32>
    %802 = mhlo.add %797, %801 : tensor<512xf32>
    %803 = "mhlo.get_tuple_element"(%120) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %804 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %805 = "mhlo.broadcast_in_dim"(%804) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %806 = mhlo.multiply %803, %805 : tensor<512xf32>
    %807 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %808 = mhlo.subtract %807, %804 : tensor<f32>
    %809 = "mhlo.broadcast_in_dim"(%808) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %810 = mhlo.multiply %arg86, %809 : tensor<512xf32>
    %811 = mhlo.add %806, %810 : tensor<512xf32>
    %812 = mhlo.constant dense<0> : tensor<i64>
    %813 = mhlo.constant dense<1> : tensor<i64>
    %814 = mhlo.constant dense<1> : tensor<i64>
    %815 = call @aten.mul.1276(%813, %814) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %816 = call @aten.add.1282(%812, %815) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %817 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %818 = "mhlo.broadcast_in_dim"(%817) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %819 = mhlo.multiply %150, %818 : tensor<512xf32>
    %820 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %821 = mhlo.subtract %820, %817 : tensor<f32>
    %822 = "mhlo.broadcast_in_dim"(%821) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %823 = mhlo.multiply %arg93, %822 : tensor<512xf32>
    %824 = mhlo.add %819, %823 : tensor<512xf32>
    %825 = "mhlo.get_tuple_element"(%129) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %826 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %827 = "mhlo.broadcast_in_dim"(%826) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %828 = mhlo.multiply %825, %827 : tensor<512xf32>
    %829 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %830 = mhlo.subtract %829, %826 : tensor<f32>
    %831 = "mhlo.broadcast_in_dim"(%830) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %832 = mhlo.multiply %arg92, %831 : tensor<512xf32>
    %833 = mhlo.add %828, %832 : tensor<512xf32>
    %834 = mhlo.constant dense<0> : tensor<i64>
    %835 = mhlo.constant dense<1> : tensor<i64>
    %836 = mhlo.constant dense<1> : tensor<i64>
    %837 = call @aten.mul.1276(%835, %836) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %838 = call @aten.add.1282(%834, %837) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %839 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %840 = "mhlo.broadcast_in_dim"(%839) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %841 = mhlo.multiply %142, %840 : tensor<512xf32>
    %842 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %843 = mhlo.subtract %842, %839 : tensor<f32>
    %844 = "mhlo.broadcast_in_dim"(%843) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %845 = mhlo.multiply %arg98, %844 : tensor<512xf32>
    %846 = mhlo.add %841, %845 : tensor<512xf32>
    %847 = "mhlo.get_tuple_element"(%134) {index = 2 : i32} : (tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>) -> tensor<512xf32>
    %848 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %849 = "mhlo.broadcast_in_dim"(%848) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %850 = mhlo.multiply %847, %849 : tensor<512xf32>
    %851 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %852 = mhlo.subtract %851, %848 : tensor<f32>
    %853 = "mhlo.broadcast_in_dim"(%852) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %854 = mhlo.multiply %arg97, %853 : tensor<512xf32>
    %855 = mhlo.add %850, %854 : tensor<512xf32>
    %856 = mhlo.constant dense<0> : tensor<i64>
    %857 = mhlo.constant dense<1> : tensor<i64>
    %858 = mhlo.constant dense<1> : tensor<i64>
    %859 = call @aten.mul.1276(%857, %858) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %860 = call @aten.add.1282(%856, %859) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %861 = call @aten.view.1222(%411) : (tensor<32x512x1x1xf16>) -> tensor<32x512xf16>
    %862 = call @aten.permute.618(%8) {minor_to_major = dense<[0, 1]> : tensor<2xindex>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    %863 = "mhlo.convert"(%arg103) : (tensor<1000xf32>) -> tensor<1000xf16>
    %864 = call @aten.addmm.1690(%861, %862, %863) : (tensor<32x512xf16>, tensor<512x1000xf16>, tensor<1000xf16>) -> tensor<32x1000xf16>
    %865 = "mhlo.tuple"(%332, %333, %334, %336, %337, %338, %340, %341, %342, %344, %345, %346, %348, %349, %350, %352, %353, %354, %356, %357, %358, %360, %361, %362, %364, %365, %366, %368, %369, %370, %372, %373, %374, %376, %377, %378, %380, %381, %382, %384, %385, %386, %388, %389, %390, %392, %393, %394, %396, %397, %398, %400, %401, %402, %404, %405, %406, %408, %409, %410, %416, %420, %428, %437, %442, %450, %459, %464, %472, %481, %486, %494, %503, %508, %516, %525, %530, %538, %547, %552, %560, %569, %574, %582, %591, %596, %604, %613, %618, %626, %635, %640, %648, %657, %662, %670, %679, %684, %692, %701, %706, %714, %723, %728, %736, %745, %750, %758, %767, %772, %780, %789, %794, %802, %811, %816, %824, %833, %838, %846, %855, %860, %864) : (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1000x512xf32>, tensor<1000xf32>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<64xf32>, tensor<64xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<128xf32>, tensor<128xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<256xf32>, tensor<256xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<512xf32>, tensor<512xf32>, tensor<i64>, tensor<32x1000xf16>) -> !tuple
    return %865 : !tuple
  }
  func private @aten.convolution_overrideable.9(%arg0: tensor<32x3x224x224xf16>, %arg1: tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>) -> tensor<32x64x112x112xf16>
    return %0 : tensor<32x64x112x112xf16>
  }
  func private @aten.native_batch_norm.14(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %8 = mhlo.add %1#2, %7 : tensor<64xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func private @aten.relu.36(%arg0: tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x112x112xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x64x112x112xf16>
    return %2 : tensor<32x64x112x112xf16>
  }
  func private @aten.max_pool2d.121(%arg0: tensor<32x64x112x112xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>> {
    %0 = mhlo.constant dense<0> : tensor<ui32>
    %1 = mhlo.constant dense<6422528> : tensor<ui32>
    %2 = mhlo.constant dense<0xFC00> : tensor<f16>
    %3 = "mhlo.pad"(%arg0, %2) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
    %4 = mhlo.constant dense<0xFC00> : tensor<f16>
    %5 = "mhlo.reduce_window"(%3, %4) ( {
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):  // no predecessors
      %23 = mhlo.maximum %arg1, %arg2 : tensor<f16>
      "mhlo.return"(%23) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
    %6 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<12544xui32>
    %7 = "mhlo.reshape"(%6) : (tensor<12544xui32>) -> tensor<112x112xui32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<112x112xui32>) -> tensor<32x64x112x112xui32>
    %9 = mhlo.constant dense<4294967295> : tensor<ui32>
    %10 = "mhlo.pad"(%8, %9) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xui32>, tensor<ui32>) -> tensor<32x64x114x114xui32>
    %11 = mhlo.constant dense<0> : tensor<ui32>
    %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<6422528xui32>
    %13 = "mhlo.tuple"(%0, %1, %3, %5, %10, %12) : (tensor<ui32>, tensor<ui32>, tensor<32x64x114x114xf16>, tensor<32x64x56x56xf16>, tensor<32x64x114x114xui32>, tensor<6422528xui32>) -> tuple<tensor<ui32>, tensor<ui32>, tensor<32x64x114x114xf16>, tensor<32x64x56x56xf16>, tensor<32x64x114x114xui32>, tensor<6422528xui32>>
    %14:6 = "mhlo.while"(%0, %1, %3, %5, %10, %12) ( {
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>, %arg3: tensor<32x64x114x114xf16>, %arg4: tensor<32x64x56x56xf16>, %arg5: tensor<32x64x114x114xui32>, %arg6: tensor<6422528xui32>):  // no predecessors
      %23 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "LT"} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
      "mhlo.return"(%23) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>, %arg3: tensor<32x64x114x114xf16>, %arg4: tensor<32x64x56x56xf16>, %arg5: tensor<32x64x114x114xui32>, %arg6: tensor<6422528xui32>):  // no predecessors
      %24 = mhlo.constant dense<200704> : tensor<ui32>
      %25 = mhlo.remainder %arg1, %24 : tensor<ui32>
      %26 = mhlo.constant dense<3136> : tensor<ui32>
      %27 = mhlo.remainder %25, %26 : tensor<ui32>
      %28 = mhlo.constant dense<56> : tensor<ui32>
      %29 = mhlo.remainder %27, %28 : tensor<ui32>
      %30 = mhlo.constant dense<1> : tensor<ui32>
      %31 = mhlo.remainder %29, %30 : tensor<ui32>
      %32 = mhlo.constant dense<1> : tensor<ui32>
      %33 = mhlo.add %arg1, %32 : tensor<ui32>
      %39 = mhlo.constant dense<1> : tensor<ui32>
      %40 = mhlo.divide %arg1, %24 : tensor<ui32>
      %41 = mhlo.multiply %39, %40 : tensor<ui32>
      %42 = mhlo.constant dense<1> : tensor<ui32>
      %43 = mhlo.divide %25, %26 : tensor<ui32>
      %44 = mhlo.multiply %42, %43 : tensor<ui32>
      %45 = mhlo.constant dense<2> : tensor<ui32>
      %46 = mhlo.divide %27, %28 : tensor<ui32>
      %47 = mhlo.multiply %45, %46 : tensor<ui32>
      %48 = mhlo.constant dense<2> : tensor<ui32>
      %49 = mhlo.divide %29, %30 : tensor<ui32>
      %50 = mhlo.multiply %48, %49 : tensor<ui32>
      %51 = "mhlo.dynamic-slice"(%arg3, %41, %44, %47, %50) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xf16>
      %52 = "mhlo.dynamic-slice"(%arg4, %40, %43, %46, %49) {slice_sizes = dense<1> : tensor<4xi64>} : (tensor<32x64x56x56xf16>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x1x1xf16>
      %53 = mhlo.constant dense<0xFC00> : tensor<f16>
      %54 = "mhlo.select_and_scatter"(%51, %52, %53) ( {
      ^bb0(%arg7: tensor<f16>, %arg8: tensor<f16>):  // no predecessors
        %65 = "mhlo.compare"(%arg7, %arg8) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
        "mhlo.return"(%65) : (tensor<i1>) -> ()
      },  {
      ^bb0(%arg7: tensor<f16>, %arg8: tensor<f16>):  // no predecessors
        %65 = mhlo.maximum %arg7, %arg8 : tensor<f16>
        "mhlo.return"(%65) : (tensor<f16>) -> ()
      }) {padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<f16>) -> tensor<1x1x3x3xf16>
      %55 = "mhlo.broadcast_in_dim"(%53) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<1x1x3x3xf16>
      %56 = "mhlo.compare"(%54, %55) {comparison_direction = "NE"} : (tensor<1x1x3x3xf16>, tensor<1x1x3x3xf16>) -> tensor<1x1x3x3xi1>
      %57 = "mhlo.dynamic-slice"(%arg5, %41, %44, %47, %50) {slice_sizes = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<32x64x114x114xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<1x1x3x3xui32>
      %58 = mhlo.constant dense<4294967295> : tensor<ui32>
      %59 = "mhlo.broadcast_in_dim"(%58) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<ui32>) -> tensor<1x1x3x3xui32>
      %60 = "mhlo.select"(%56, %57, %59) : (tensor<1x1x3x3xi1>, tensor<1x1x3x3xui32>, tensor<1x1x3x3xui32>) -> tensor<1x1x3x3xui32>
      %61 = "mhlo.reduce_window"(%60, %58) ( {
      ^bb0(%arg7: tensor<ui32>, %arg8: tensor<ui32>):  // no predecessors
        %65 = mhlo.minimum %arg7, %arg8 : tensor<ui32>
        "mhlo.return"(%65) : (tensor<ui32>) -> ()
      }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x1x3x3xui32>, tensor<ui32>) -> tensor<1x1x1x1xui32>
      %62 = "mhlo.reshape"(%61) : (tensor<1x1x1x1xui32>) -> tensor<1xui32>
      %63 = "mhlo.dynamic-update-slice"(%arg6, %62, %arg1) : (tensor<6422528xui32>, tensor<1xui32>, tensor<ui32>) -> tensor<6422528xui32>
      "mhlo.return"(%33, %arg2, %arg3, %arg4, %arg5, %63) : (tensor<ui32>, tensor<ui32>, tensor<32x64x114x114xf16>, tensor<32x64x56x56xf16>, tensor<32x64x114x114xui32>, tensor<6422528xui32>) -> ()
    }) : (tensor<ui32>, tensor<ui32>, tensor<32x64x114x114xf16>, tensor<32x64x56x56xf16>, tensor<32x64x114x114xui32>, tensor<6422528xui32>) -> (tensor<ui32>, tensor<ui32>, tensor<32x64x114x114xf16>, tensor<32x64x56x56xf16>, tensor<32x64x114x114xui32>, tensor<6422528xui32>)
    %21 = "mhlo.reshape"(%14#5) : (tensor<6422528xui32>) -> tensor<32x64x56x56xui32>
    %22 = "mhlo.tuple"(%5, %21) : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>) -> tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>>
    return %22 : tuple<tensor<32x64x56x56xf16>, tensor<32x64x56x56xui32>>
  }
  func private @aten.permute.618(%arg0: tensor<1000x512xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1000x512xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func private @aten.permute.622(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func private @aten.mm.627(%arg0: tensor<32x1000xf16>, %arg1: tensor<1000x512xf16>) -> tensor<32x512xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x1000xf16>, tensor<1000x512xf16>) -> tensor<32x512xf16>
    return %0 : tensor<32x512xf16>
  }
  func private @aten.view.632(%arg0: tensor<32x512xf16>) -> tensor<32x512x1x1xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<32x512xf16>) -> tensor<32x512x1x1xf16>
    return %0 : tensor<32x512x1x1xf16>
  }
  func private @aten.expand.636(%arg0: tensor<32x512x1x1xf16>) -> tensor<32x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<32x512x1x1xf16>) -> tensor<32x512x1x1xf16>
    %1 = "mhlo.reshape"(%0) : (tensor<32x512x1x1xf16>) -> tensor<32x512xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<32x512xf16>) -> tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @aten.div.642(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<f16>) -> tensor<32x512x7x7xf16> {
    %0 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %1 = mhlo.divide %arg0, %0 : tensor<32x512x7x7xf16>
    return %1 : tensor<32x512x7x7xf16>
  }
  func private @aten.convolution_overrideable.153(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
    return %0 : tensor<32x64x56x56xf16>
  }
  func private @aten.native_batch_norm.158(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %8 = mhlo.add %1#2, %7 : tensor<64xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<64xf32>) -> tensor<64xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
    return %10 : tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>>
  }
  func private @aten.relu.180(%arg0: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x64x56x56xf16>
    return %2 : tensor<32x64x56x56xf16>
  }
  func private @aten.expand.43(%arg0: tensor<f16>) -> tensor<32x64x56x56xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x56x56xf16>
    return %3 : tensor<32x64x56x56xf16>
  }
  func private @aten.mul.200(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<32x64x56x56xf16>
    return %0 : tensor<32x64x56x56xf16>
  }
  func private @aten.add.205(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x64x56x56xf16>
    return %0 : tensor<32x64x56x56xf16>
  }
  func private @aten.convolution_overrideable.251(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>) -> tensor<32x128x28x28xf16>
    return %0 : tensor<32x128x28x28xf16>
  }
  func private @aten.native_batch_norm.256(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %8 = mhlo.add %1#2, %7 : tensor<128xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<128xf32>) -> tensor<128xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
    return %10 : tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>>
  }
  func private @aten.relu.278(%arg0: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x128x28x28xf16>
    return %2 : tensor<32x128x28x28xf16>
  }
  func private @aten.convolution_overrideable.290(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>) -> tensor<32x128x28x28xf16>
    return %0 : tensor<32x128x28x28xf16>
  }
  func private @aten.convolution_overrideable.314(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>) -> tensor<32x128x28x28xf16>
    return %0 : tensor<32x128x28x28xf16>
  }
  func private @aten.expand.301(%arg0: tensor<f16>) -> tensor<32x128x28x28xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x128x28x28xf16>
    return %3 : tensor<32x128x28x28xf16>
  }
  func private @aten.mul.324(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<32x128x28x28xf16>
    return %0 : tensor<32x128x28x28xf16>
  }
  func private @aten.add.329(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x128x28x28xf16>
    return %0 : tensor<32x128x28x28xf16>
  }
  func private @aten.convolution_overrideable.375(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>) -> tensor<32x256x14x14xf16>
    return %0 : tensor<32x256x14x14xf16>
  }
  func private @aten.native_batch_norm.380(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %8 = mhlo.add %1#2, %7 : tensor<256xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<256xf32>) -> tensor<256xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
    return %10 : tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>>
  }
  func private @aten.relu.402(%arg0: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x256x14x14xf16>
    return %2 : tensor<32x256x14x14xf16>
  }
  func private @aten.convolution_overrideable.414(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x14x14xf16>
    return %0 : tensor<32x256x14x14xf16>
  }
  func private @aten.convolution_overrideable.438(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>) -> tensor<32x256x14x14xf16>
    return %0 : tensor<32x256x14x14xf16>
  }
  func private @aten.expand.425(%arg0: tensor<f16>) -> tensor<32x256x14x14xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x256x14x14xf16>
    return %3 : tensor<32x256x14x14xf16>
  }
  func private @aten.mul.448(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<32x256x14x14xf16>
    return %0 : tensor<32x256x14x14xf16>
  }
  func private @aten.add.453(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x256x14x14xf16>
    return %0 : tensor<32x256x14x14xf16>
  }
  func private @aten.convolution_overrideable.499(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>) -> tensor<32x512x7x7xf16>
    return %0 : tensor<32x512x7x7xf16>
  }
  func private @aten.native_batch_norm.504(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>> {
    %0 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %3 = "mhlo.convert"(%1#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %6 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %7 = "mhlo.broadcast_in_dim"(%6) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %8 = mhlo.add %1#2, %7 : tensor<512xf32>
    %9 = "mhlo.rsqrt"(%8) : (tensor<512xf32>) -> tensor<512xf32>
    %10 = "mhlo.tuple"(%3, %1#1, %1#2, %9) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
    return %10 : tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>>
  }
  func private @aten.relu.526(%arg0: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x512x7x7xf16>
    return %2 : tensor<32x512x7x7xf16>
  }
  func private @aten.convolution_overrideable.538(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>) -> tensor<32x512x7x7xf16>
    return %0 : tensor<32x512x7x7xf16>
  }
  func private @aten.convolution_overrideable.562(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) -> tensor<32x512x7x7xf16>
    return %0 : tensor<32x512x7x7xf16>
  }
  func private @aten.expand.549(%arg0: tensor<f16>) -> tensor<32x512x7x7xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<f16>) -> tensor<1x1x1x1xf16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
    %2 = "mhlo.reshape"(%1) : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %3 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512x7x7xf16>
    return %3 : tensor<32x512x7x7xf16>
  }
  func private @aten.mul.572(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<32x512x7x7xf16>
    return %0 : tensor<32x512x7x7xf16>
  }
  func private @aten.add.577(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.add %arg0, %arg1 : tensor<32x512x7x7xf16>
    return %0 : tensor<32x512x7x7xf16>
  }
  func private @aten.threshold_backward.648(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512x7x7xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<32x512x7x7xi1>, tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf16>
    return %5 : tensor<32x512x7x7xf16>
  }
  func private @aten.native_batch_norm_backward.658(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<512xf32>
    %4 = mhlo.multiply %3, %3 : tensor<512xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<512xf32>
    %7 = mhlo.subtract %4, %6 : tensor<512xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x512x7x7xf16>) -> tensor<32x512x7x7xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<32x512x7x7xf32>) -> (tensor<32x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>) -> tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
    return %14 : tuple<tensor<32x512x7x7xf16>, tensor<512xf32>, tensor<512xf32>>
  }
  func private @aten.convolution_backward_overrideable.687(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x512x7x7xf16>, %arg2: tensor<512x512x3x3xf16>) -> tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x512x3x3xf16>) -> tensor<3x3x512x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x512x512xf16>) -> tensor<3x3x512x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x512x512xf16>) -> tensor<32x512x7x7xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x512x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x512x512xf16>) -> tensor<512x512x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>) -> tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<32x512x7x7xf16>, tensor<512x512x3x3xf16>, tensor<512xf16>>
  }
  func private @aten.convolution_backward_overrideable.732(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<512x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x3x3xf16>) -> tensor<3x3x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x512xf16>) -> tensor<3x3x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<3x3x256x512xf16>) -> tensor<32x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<3x3x256x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x512xf16>) -> tensor<512x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<32x256x14x14xf16>, tensor<512x256x3x3xf16>, tensor<512xf16>>
  }
  func private @aten.convolution_backward_overrideable.757(%arg0: tensor<32x512x7x7xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<512x256x1x1xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x256x512xf16>) -> tensor<1x1x256x512xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<512xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
    return %7 : tuple<tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>>
  }
  func private @aten.threshold_backward.774(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x256x14x14xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<32x256x14x14xi1>, tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf16>
    return %5 : tensor<32x256x14x14xf16>
  }
  func private @aten.native_batch_norm_backward.784(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<256xf32>
    %4 = mhlo.multiply %3, %3 : tensor<256xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
    %7 = mhlo.subtract %4, %6 : tensor<256xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) -> tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
    return %14 : tuple<tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>>
  }
  func private @aten.convolution_backward_overrideable.813(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<256x256x3x3xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x256x3x3xf16>) -> tensor<3x3x256x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x256x256xf16>) -> tensor<3x3x256x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x256x256xf16>) -> tensor<32x256x14x14xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x256x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x256x256xf16>) -> tensor<256x256x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>) -> tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<32x256x14x14xf16>, tensor<256x256x3x3xf16>, tensor<256xf16>>
  }
  func private @aten.convolution_backward_overrideable.858(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<256x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x3x3xf16>) -> tensor<3x3x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x256xf16>) -> tensor<3x3x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<3x3x128x256xf16>) -> tensor<32x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<3x3x128x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x256xf16>) -> tensor<256x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<32x128x28x28xf16>, tensor<256x128x3x3xf16>, tensor<256xf16>>
  }
  func private @aten.convolution_backward_overrideable.883(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<256x128x1x1xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<256x128x1x1xf16>) -> tensor<1x1x128x256xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x128x256xf16>) -> tensor<1x1x128x256xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x256x14x14xf16>, tensor<1x1x128x256xf16>) -> tensor<32x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x256x14x14xf16>) -> tensor<1x1x128x256xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x128x256xf16>) -> tensor<256x128x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x256x14x14xf16>, tensor<f16>) -> tensor<256xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
    return %7 : tuple<tensor<32x128x28x28xf16>, tensor<256x128x1x1xf16>, tensor<256xf16>>
  }
  func private @aten.threshold_backward.900(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x128x28x28xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<32x128x28x28xi1>, tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf16>
    return %5 : tensor<32x128x28x28xf16>
  }
  func private @aten.native_batch_norm_backward.910(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<128xf32>
    %4 = mhlo.multiply %3, %3 : tensor<128xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<128xf32>
    %7 = mhlo.subtract %4, %6 : tensor<128xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x128x28x28xf16>) -> tensor<32x128x28x28xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<32x128x28x28xf32>) -> (tensor<32x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>) -> tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
    return %14 : tuple<tensor<32x128x28x28xf16>, tensor<128xf32>, tensor<128xf32>>
  }
  func private @aten.convolution_backward_overrideable.939(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x128x28x28xf16>, %arg2: tensor<128x128x3x3xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x128x3x3xf16>) -> tensor<3x3x128x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x128x128xf16>) -> tensor<3x3x128x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x128x128xf16>) -> tensor<32x128x28x28xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x128x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x128x128xf16>) -> tensor<128x128x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>) -> tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<32x128x28x28xf16>, tensor<128x128x3x3xf16>, tensor<128xf16>>
  }
  func private @aten.convolution_backward_overrideable.984(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<128x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x3x3xf16>) -> tensor<3x3x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x128xf16>) -> tensor<3x3x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<3x3x64x128xf16>) -> tensor<32x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<3x3x64x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x128xf16>) -> tensor<128x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<32x64x56x56xf16>, tensor<128x64x3x3xf16>, tensor<128xf16>>
  }
  func private @aten.convolution_backward_overrideable.1009(%arg0: tensor<32x128x28x28xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<128x64x1x1xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<128x64x1x1xf16>) -> tensor<1x1x64x128xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<1x1x64x128xf16>) -> tensor<1x1x64x128xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x128x28x28xf16>, tensor<1x1x64x128xf16>) -> tensor<32x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x128x28x28xf16>) -> tensor<1x1x64x128xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x64x128xf16>) -> tensor<128x64x1x1xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x128x28x28xf16>, tensor<f16>) -> tensor<128xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
    return %7 : tuple<tensor<32x64x56x56xf16>, tensor<128x64x1x1xf16>, tensor<128xf16>>
  }
  func private @aten.threshold_backward.1026(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x56x56xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<32x64x56x56xi1>, tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf16>
    return %5 : tensor<32x64x56x56xf16>
  }
  func private @aten.native_batch_norm_backward.1036(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x64x56x56xf16>) -> tensor<32x64x56x56xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x56x56xf32>) -> (tensor<32x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
    return %14 : tuple<tensor<32x64x56x56xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func private @aten.convolution_backward_overrideable.1065(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<64x64x3x3xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<64xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>) -> tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>>
  }
  func private @aten.max_pool2d_with_indices_backward.1120(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.select_and_scatter"(%arg1, %arg0, %0) ( {
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
      %2 = "mhlo.compare"(%arg2, %arg3) {comparison_direction = "GE"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
      %2 = mhlo.add %arg2, %arg3 : tensor<f16>
      "mhlo.return"(%2) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
    return %1 : tensor<32x64x112x112xf16>
  }
  func private @aten.threshold_backward.1126(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x112x112xf16>
    %2 = "mhlo.compare"(%arg1, %1) {comparison_direction = "GT"} : (tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x64x112x112xf16>
    %5 = "mhlo.select"(%2, %arg0, %4) : (tensor<32x64x112x112xi1>, tensor<32x64x112x112xf16>, tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf16>
    return %5 : tensor<32x64x112x112xf16>
  }
  func private @aten.native_batch_norm_backward.1136(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<32x64x112x112xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>> {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %3 = mhlo.divide %2, %arg4 : tensor<64xf32>
    %4 = mhlo.multiply %3, %3 : tensor<64xf32>
    %5 = mhlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<64xf32>
    %7 = mhlo.subtract %4, %6 : tensor<64xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x64x112x112xf16>) -> tensor<32x64x112x112xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<32x64x112x112xf32>) -> (tensor<32x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf16>
    %14 = "mhlo.tuple"(%11, %9#1, %9#2) : (tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>) -> tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
    return %14 : tuple<tensor<32x64x112x112xf16>, tensor<64xf32>, tensor<64xf32>>
  }
  func private @aten.convolution_backward_overrideable.1165(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<32x3x224x224xf16>, %arg2: tensor<64x3x7x7xf16>) -> tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>> {
    %0 = "mhlo.transpose"(%arg2) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x3x7x7xf16>) -> tensor<7x7x3x64xf16>
    %1 = "mhlo.reverse"(%0) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<7x7x3x64xf16>) -> tensor<7x7x3x64xf16>
    %2 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[3, 4], [3, 4]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x64x112x112xf16>, tensor<7x7x3x64xf16>) -> tensor<32x3x224x224xf16>
    %3 = mhlo.convolution(%arg1, %arg0) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x3x224x224xf16>, tensor<32x64x112x112xf16>) -> tensor<7x7x3x64xf16>
    %4 = "mhlo.transpose"(%3) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<7x7x3x64xf16>) -> tensor<64x3x7x7xf16>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %6 = "mhlo.reduce"(%arg0, %5) ( {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %8 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%8) : (tensor<f16>) -> ()
    }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<64xf16>
    %7 = "mhlo.tuple"(%2, %4, %6) : (tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>) -> tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
    return %7 : tuple<tensor<32x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>>
  }
  func private @aten.mean.1205(%arg0: tensor<32x512x7x7xf16>) -> tensor<32x512x1x1xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):  // no predecessors
      %14 = mhlo.add %arg1, %arg2 : tensor<f16>
      "mhlo.return"(%14) : (tensor<f16>) -> ()
    }) {dimensions = dense<[3, 2]> : tensor<2xi64>} : (tensor<32x512x7x7xf16>, tensor<f16>) -> tensor<32x512xf16>
    %2 = mhlo.constant dense<49> : tensor<i64>
    %3 = mhlo.constant dense<0> : tensor<i64>
    %4 = "mhlo.compare"(%2, %3) {comparison_direction = "NE"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = mhlo.constant dense<1.000000e+00> : tensor<f16>
    %6 = "mhlo.convert"(%2) : (tensor<i64>) -> tensor<f16>
    %7 = mhlo.divide %5, %6 : tensor<f16>
    %8 = mhlo.constant dense<0x7E00> : tensor<f16>
    %9 = "mhlo.select"(%4, %7, %8) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>) -> tensor<32x512xf16>
    %11 = mhlo.multiply %1, %10 : tensor<32x512xf16>
    %12 = "mhlo.reshape"(%11) : (tensor<32x512xf16>) -> tensor<32x512x1x1xf16>
    %13 = "mhlo.convert"(%12) : (tensor<32x512x1x1xf16>) -> tensor<32x512x1x1xf16>
    return %13 : tensor<32x512x1x1xf16>
  }
  func private @aten.view.1222(%arg0: tensor<32x512x1x1xf16>) -> tensor<32x512xf16> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<32x512x1x1xf16>) -> tensor<32x512xf16>
    return %0 : tensor<32x512xf16>
  }
  func private @aten.permute.1226(%arg0: tensor<32x512xf16>) -> tensor<512x32xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<32x512xf16>) -> tensor<512x32xf16>
    return %0 : tensor<512x32xf16>
  }
  func private @aten.mm.1230(%arg0: tensor<512x32xf16>, %arg1: tensor<32x1000xf16>) -> tensor<512x1000xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<512x32xf16>, tensor<32x1000xf16>) -> tensor<512x1000xf16>
    return %0 : tensor<512x1000xf16>
  }
  func private @aten.permute.1235(%arg0: tensor<512x1000xf16>) -> tensor<1000x512xf16> {
    %0 = "mhlo.transpose"(%arg0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x1000xf16>) -> tensor<1000x512xf16>
    return %0 : tensor<1000x512xf16>
  }
  func private @aten.sum.1244(%arg0: tensor<32x1000xf16>) -> tensor<1x1000xf32> {
    %0 = mhlo.constant dense<32> : tensor<i64>
    %1 = "mhlo.convert"(%arg0) : (tensor<32x1000xf16>) -> tensor<32x1000xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.reduce"(%1, %2) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %5 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%5) : (tensor<f32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<32x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %4 = "mhlo.reshape"(%3) : (tensor<1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
  }
  func private @aten.view.1252(%arg0: tensor<1x1000xf32>) -> tensor<1000xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x1000xf32>) -> tensor<1000xf32>
    return %0 : tensor<1000xf32>
  }
  func private @aten.mul.1276(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = mhlo.multiply %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
  func private @aten.add.1282(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = mhlo.add %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
  func private @aten.addmm.1690(%arg0: tensor<32x512xf16>, %arg1: tensor<512x1000xf16>, %arg2: tensor<1000xf16>) -> tensor<32x1000xf16> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<32x512xf16>, tensor<512x1000xf16>) -> tensor<32x1000xf16>
    %1 = "mhlo.reshape"(%arg2) : (tensor<1000xf16>) -> tensor<1x1000xf16>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1000xf16>) -> tensor<1x1000xf16>
    %3 = "mhlo.reshape"(%2) : (tensor<1x1000xf16>) -> tensor<1000xf16>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf16>) -> tensor<32x1000xf16>
    %5 = mhlo.add %0, %4 : tensor<32x1000xf16>
    return %5 : tensor<32x1000xf16>
  }
}
