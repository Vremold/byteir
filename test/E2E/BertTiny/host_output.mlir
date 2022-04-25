// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %0 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%0) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    %1 = memref.alloc() : memref<2x128xf32>
    byre.compute @FillOp(%1) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @FillOp(%2) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32>
    %3 = memref.alloc() : memref<1x128xi64>
    byre.compute @AliasOp(%arg2, %3) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %4 = memref.alloc() : memref<128xi64>
    byre.compute @AliasOp(%arg2, %4) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<128xi64>
    %5 = memref.alloc() : memref<1x128xi64>
    byre.compute @AliasOp(%arg3, %5) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %6 = memref.alloc() : memref<256xi64>
    %7 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg1, %7) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0"} : memref<2x128xi64>, memref<256xi1>
    byre.compute @AliasOp(%arg1, %6) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %8 = memref.alloc() : memref<256xui32>
    %9 = memref.alloc() : memref<256x1xi64>
    %10 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg0, %8, %9, %10) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %11 = memref.alloc() : memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %8, %11) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %12 = memref.alloc() : memref<256xui32>
    %13 = memref.alloc() : memref<256x1xi64>
    %14 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%4, %12, %13, %14) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %15 = memref.alloc() : memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %12, %15) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %16 = memref.alloc() : memref<128xui32>
    %17 = memref.alloc() : memref<128x1xi64>
    %18 = memref.alloc() : memref<128xi1>
    byre.compute @PTXOp(%5, %16, %17, %18) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    %19 = memref.alloc() : memref<128x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %16, %19) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %20 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%11, %15, %19, %20) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    %21 = memref.alloc() : memref<2x128x128xf32>
    %22 = memref.alloc() : memref<256xf32>
    %23 = memref.alloc() : memref<256xf32>
    byre.compute @ftv4.layernorm(%20, %arg7, %arg8, %21, %22, %23) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %24 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%21, %arg9, %arg10, %24) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%21, %arg11, %arg12, %25) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %26 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%24, %25, %26) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %27 = memref.alloc() : memref<2x2x128x128xf32>
    %28 = memref.alloc() : memref<2x2x128x128xf32>
    %29 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%26, %2, %27, %28, %29) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %30 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%21, %arg13, %arg14, %30) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %31 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%27, %30, %31) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %32 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%31, %32) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%32, %33) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear(%33, %arg15, %arg16, %34) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %35 = memref.alloc() : memref<2x128x128xf32>
    %36 = memref.alloc() : memref<256xf32>
    %37 = memref.alloc() : memref<256xf32>
    %38 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%34, %arg17, %arg18, %21, %35, %36, %37, %38) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %39 = memref.alloc() : memref<2x128x512xf32>
    %40 = memref.alloc() : memref<2x128x512xf32>
    %41 = memref.alloc() : memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%35, %arg19, %arg20, %39, %40, %41) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %42 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear(%39, %arg21, %arg22, %42) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %43 = memref.alloc() : memref<2x128x128xf32>
    %44 = memref.alloc() : memref<256xf32>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%42, %arg23, %arg24, %35, %43, %44, %45, %46) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %47 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%43, %arg25, %arg26, %47) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %48 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%43, %arg27, %arg28, %48) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %49 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%47, %48, %49) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %50 = memref.alloc() : memref<2x2x128x128xf32>
    %51 = memref.alloc() : memref<2x2x128x128xf32>
    %52 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%49, %2, %50, %51, %52) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %53 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%43, %arg29, %arg30, %53) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%50, %53, %54) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %55 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%54, %55) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %56 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %56) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    %57 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear(%56, %arg31, %arg32, %57) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %58 = memref.alloc() : memref<2x128x128xf32>
    %59 = memref.alloc() : memref<256xf32>
    %60 = memref.alloc() : memref<256xf32>
    %61 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%57, %arg33, %arg34, %43, %58, %59, %60, %61) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %62 = memref.alloc() : memref<2x128x512xf32>
    %63 = memref.alloc() : memref<2x128x512xf32>
    %64 = memref.alloc() : memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%58, %arg35, %arg36, %62, %63, %64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %65 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear(%62, %arg37, %arg38, %65) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<256xf32>
    %68 = memref.alloc() : memref<256xf32>
    %69 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%65, %arg39, %arg40, %58, %66, %67, %68, %69) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %70 = memref.alloc() : memref<2x128x128xf32>
    %71 = memref.alloc() : memref<2x128x128xf32>
    %72 = memref.alloc() : memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%66, %arg41, %arg42, %70, %71, %72) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    %73 = memref.alloc() : memref<2x128x128xf32>
    %74 = memref.alloc() : memref<256xf32>
    %75 = memref.alloc() : memref<256xf32>
    byre.compute @ftv4.layernorm(%70, %arg43, %arg44, %73, %74, %75) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %76 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%73, %76) {offset = 0 : i32} : memref<2x128x128xf32>, memref<256x128xf32>
    %77 = memref.alloc() : memref<256x30522xf32>
    byre.compute @MatmulOpf32f32f32(%76, %arg4, %77) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    %78 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%77, %arg45, %arg46, %78) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32, 2 : i32], kernel_name = "Unknown5"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>, memref<256x30522xf32>
    %79 = memref.alloc() : memref<256xf32>
    byre.compute @ReduceMaxOpf32f32(%78, %79) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %80 = memref.alloc() : memref<256x30522xf32>
    %81 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%79, %78, %80, %81) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %82 = memref.alloc() : memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%81, %82) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %83 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%82, %83) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7"} : memref<256xf32>, memref<256xf32>
    %84 = memref.alloc() : memref<256x30522xf32>
    %85 = memref.alloc() : memref<256x30522xf32>
    %86 = memref.alloc() : memref<256x30522xf32>
    %87 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%83, %80, %6, %7, %84, %85, %86, %87) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8"} : memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %88 = memref.alloc() : memref<f32>
    byre.compute @ReduceSumOpf32f32(%84, %88) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %89 = memref.alloc() : memref<f32>
    byre.compute @ReduceSumOpf32f32(%84, %89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %90 = memref.alloc() : memref<f32>
    byre.compute @PTXOp(%89, %90) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown9"} : memref<f32>, memref<f32>
    %91 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%90, %86, %91) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown10"} : memref<f32>, memref<256x30522xf32>, memref<256x30522xf32>
    %92 = memref.alloc() : memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%91, %92) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %93 = memref.alloc() : memref<f32>
    byre.compute @ReduceSumOpf32f32(%85, %93) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%93, %88, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown11"} : memref<f32>, memref<f32>, memref<f32>
    %94 = memref.alloc() : memref<256x30522xf32>
    %95 = memref.alloc() : memref<2x128x30522xf32>
    byre.compute @PTXOp(%92, %87, %91, %94, %95) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<2x128x30522xf32>
    %96 = memref.alloc() : memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%76, %94, %96) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    %97 = memref.alloc() : memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%94, %arg4, %97) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    %98 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%97, %98) {offset = 0 : i32} : memref<256x128xf32>, memref<2x128x128xf32>
    %99 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%98, %70, %arg43, %74, %75, %99, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %100 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%99, %66, %arg41, %71, %72, %100, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %101 = memref.alloc() : memref<2x128x128xf32>
    %102 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%100, %69, %arg39, %67, %68, %101, %arg83, %arg84, %102) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %103 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%101, %62, %arg37, %103, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %104 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%103, %58, %arg35, %63, %64, %104, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%102, %104, %105) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %106 = memref.alloc() : memref<2x128x128xf32>
    %107 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%105, %61, %arg33, %59, %60, %106, %arg77, %arg78, %107) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %108 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%106, %56, %arg31, %108, %arg75, %arg76) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %109 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%108, %109) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    %110 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%109, %110) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %111 = memref.alloc() : memref<2x2x128x128xf32>
    %112 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%110, %50, %53, %111, %112) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %113 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%111, %50, %52, %113) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %114 = memref.alloc() : memref<2x2x128x64xf32>
    %115 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%113, %47, %48, %114, %115) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %116 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%114, %43, %arg25, %116, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %117 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%112, %43, %arg29, %117, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %118 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%115, %43, %arg27, %118, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %119 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%107, %116, %117, %118, %119) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %120 = memref.alloc() : memref<2x128x128xf32>
    %121 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%119, %46, %arg23, %44, %45, %120, %arg67, %arg68, %121) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %122 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%120, %39, %arg21, %122, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%122, %35, %arg19, %40, %41, %123, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %124 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%121, %123, %124) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %125 = memref.alloc() : memref<2x128x128xf32>
    %126 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%124, %38, %arg17, %36, %37, %125, %arg61, %arg62, %126) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %127 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%125, %33, %arg15, %127, %arg59, %arg60) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %128 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%127, %128) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    %129 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%128, %129) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %130 = memref.alloc() : memref<2x2x128x128xf32>
    %131 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%129, %27, %30, %130, %131) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %132 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%130, %27, %29, %132) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %133 = memref.alloc() : memref<2x2x128x64xf32>
    %134 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%132, %24, %25, %133, %134) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %135 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%133, %21, %arg9, %135, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %136 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%131, %21, %arg13, %136, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%134, %21, %arg11, %137, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %138 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%126, %135, %136, %137, %138) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %139 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%138, %20, %arg7, %22, %23, %139, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %140 = memref.alloc() : memref<256x128xf32>
    %141 = memref.alloc() : memref<256x128xf32>
    byre.compute @PTXOp(%10, %139, %14, %140, %141) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18"} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%96, %9, %140, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%1, %13, %141, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    %142 = memref.alloc() : memref<128x128xf32>
    byre.compute @ReduceSumOpf32f32(%139, %142) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %143 = memref.alloc() : memref<128x128xf32>
    byre.compute @PTXOp(%18, %142, %143) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%0, %17, %143, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%95, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

