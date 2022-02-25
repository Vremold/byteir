// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %0 = memref.alloc() : memref<128x128xf32>
    %1 = memref.alloc() : memref<2x128x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    %3 = memref.alloc() : memref<2x128x128xf32>
    %4 = memref.alloc() : memref<2x128x128xf32>
    %5 = memref.alloc() : memref<2x2x128x64xf32>
    %6 = memref.alloc() : memref<2x2x128x64xf32>
    %7 = memref.alloc() : memref<2x2x128x128xf32>
    %8 = memref.alloc() : memref<2x2x128x64xf32>
    %9 = memref.alloc() : memref<2x2x128x128xf32>
    %10 = memref.alloc() : memref<2x2x128x64xf32>
    %11 = memref.alloc() : memref<2x128x2x64xf32>
    %12 = memref.alloc() : memref<2x128x128xf32>
    %13 = memref.alloc() : memref<2x128x128xf32>
    %14 = memref.alloc() : memref<2x128x128xf32>
    %15 = memref.alloc() : memref<2x128x128xf32>
    %16 = memref.alloc() : memref<2x128x512xf32>
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<2x128x128xf32>
    %19 = memref.alloc() : memref<2x128x128xf32>
    %20 = memref.alloc() : memref<2x128x128xf32>
    %21 = memref.alloc() : memref<2x128x128xf32>
    %22 = memref.alloc() : memref<2x2x128x64xf32>
    %23 = memref.alloc() : memref<2x2x128x64xf32>
    %24 = memref.alloc() : memref<2x2x128x128xf32>
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    %26 = memref.alloc() : memref<2x2x128x128xf32>
    %27 = memref.alloc() : memref<2x2x128x64xf32>
    %28 = memref.alloc() : memref<2x128x2x64xf32>
    %29 = memref.alloc() : memref<2x128x128xf32>
    %30 = memref.alloc() : memref<2x128x128xf32>
    %31 = memref.alloc() : memref<2x128x128xf32>
    %32 = memref.alloc() : memref<2x128x128xf32>
    %33 = memref.alloc() : memref<2x128x512xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<2x128x128xf32>
    %36 = memref.alloc() : memref<2x128x128xf32>
    %37 = memref.alloc() : memref<2x128x128xf32>
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<256x128xf32>
    %40 = memref.alloc() : memref<f32>
    %41 = memref.alloc() : memref<256xf32>
    %42 = memref.alloc() : memref<f32>
    %43 = memref.alloc() : memref<f32>
    %44 = memref.alloc() : memref<256xf32>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<256x30522xf32>
    %47 = memref.alloc() : memref<256x128xf32>
    %48 = memref.alloc() : memref<256xf32>
    %49 = memref.alloc() : memref<256xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<0xf32>
    %52 = memref.alloc() : memref<2x128x128xf32>
    %53 = memref.alloc() : memref<2x128x128xf32>
    %54 = memref.alloc() : memref<2x128x128xf32>
    %55 = memref.alloc() : memref<256xf32>
    %56 = memref.alloc() : memref<256xf32>
    %57 = memref.alloc() : memref<2x128x128xf32>
    %58 = memref.alloc() : memref<2x128x128xf32>
    %59 = memref.alloc() : memref<0xf32>
    %60 = memref.alloc() : memref<2x128x512xf32>
    %61 = memref.alloc() : memref<2x128x512xf32>
    %62 = memref.alloc() : memref<2x128x128xf32>
    %63 = memref.alloc() : memref<256xf32>
    %64 = memref.alloc() : memref<256xf32>
    %65 = memref.alloc() : memref<2x128x128xf32>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<2x128x2x64xf32>
    %69 = memref.alloc() : memref<2x2x128x64xf32>
    %70 = memref.alloc() : memref<2x2x128x64xf32>
    %71 = memref.alloc() : memref<2x2x128x128xui8>
    %72 = memref.alloc() : memref<2x2x128x128xf32>
    %73 = memref.alloc() : memref<2x2x128x128xf32>
    %74 = memref.alloc() : memref<2x2x128x128xf32>
    %75 = memref.alloc() : memref<2x2x128x64xf32>
    %76 = memref.alloc() : memref<2x2x128x64xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    %78 = memref.alloc() : memref<256xf32>
    %79 = memref.alloc() : memref<256xf32>
    %80 = memref.alloc() : memref<2x128x128xf32>
    %81 = memref.alloc() : memref<2x128x128xf32>
    %82 = memref.alloc() : memref<0xf32>
    %83 = memref.alloc() : memref<2x128x512xf32>
    %84 = memref.alloc() : memref<2x128x512xf32>
    %85 = memref.alloc() : memref<2x128x128xf32>
    %86 = memref.alloc() : memref<256xf32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<2x128x128xf32>
    %89 = memref.alloc() : memref<2x128x128xf32>
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<2x128x2x64xf32>
    %92 = memref.alloc() : memref<2x2x128x64xf32>
    %93 = memref.alloc() : memref<2x2x128x64xf32>
    %94 = memref.alloc() : memref<2x2x128x128xui8>
    %95 = memref.alloc() : memref<2x2x128x128xf32>
    %96 = memref.alloc() : memref<2x2x128x128xf32>
    %97 = memref.alloc() : memref<2x2x128x128xf32>
    %98 = memref.alloc() : memref<2x2x128x64xf32>
    %99 = memref.alloc() : memref<2x2x128x64xf32>
    %100 = memref.alloc() : memref<256xf32>
    %101 = memref.alloc() : memref<256xf32>
    %102 = memref.alloc() : memref<2x128x128xf32>
    %103 = memref.alloc() : memref<128x128xf32>
    %104 = memref.alloc() : memref<256x128xf32>
    %105 = memref.alloc() : memref<256x128xf32>
    %106 = memref.alloc() : memref<1x128xi64>
    %107 = memref.alloc() : memref<128xi64>
    %108 = memref.alloc() : memref<1x128xi64>
    %109 = memref.alloc() : memref<2x128x128xf32>
    %110 = memref.alloc() : memref<2x128xf32>
    %111 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%111) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    byre.compute @FillOp(%110) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    byre.compute @FillOp(%109) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg2, %108) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    byre.compute @AliasOp(%arg2, %107) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<128xi64>
    byre.compute @AliasOp(%arg3, %106) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %112 = memref.alloc() : memref<256xi64>
    %113 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg1, %113) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0_kernel"} : memref<2x128xi64>, memref<256xi1>
    byre.compute @AliasOp(%arg1, %112) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %114 = memref.alloc() : memref<256xui32>
    %115 = memref.alloc() : memref<256x1xi64>
    %116 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg0, %114, %115, %116) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1_kernel"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %114, %105) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %117 = memref.alloc() : memref<256xui32>
    %118 = memref.alloc() : memref<256x1xi64>
    %119 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%107, %117, %118, %119) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2_kernel"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %117, %104) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %120 = memref.alloc() : memref<128xui32>
    %121 = memref.alloc() : memref<128x1xi64>
    %122 = memref.alloc() : memref<128xi1>
    byre.compute @PTXOp(%106, %120, %121, %122) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3_kernel"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %120, %103) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%105, %104, %103, %123) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4_kernel"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm(%123, %arg7, %arg8, %102, %101, %100) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @ftv4.linear_transpose(%102, %arg9, %arg10, %99) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%102, %arg11, %arg12, %98) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%99, %98, %97) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%97, %109, %96, %95, %94) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%102, %arg13, %arg14, %93) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%96, %93, %92) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%92, %91) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%91, %90) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%90, %arg15, %arg16, %89) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%89, %arg17, %arg18, %102, %88, %87, %86, %85) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%88, %arg19, %arg20, %84, %83, %82) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%84, %arg21, %arg22, %81) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%81, %arg23, %arg24, %88, %80, %79, %78, %77) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose(%80, %arg25, %arg26, %76) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%80, %arg27, %arg28, %75) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%76, %75, %74) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%74, %109, %73, %72, %71) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%80, %arg29, %arg30, %70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%73, %70, %69) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%69, %68) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%68, %67) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%67, %arg31, %arg32, %66) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%66, %arg33, %arg34, %80, %65, %64, %63, %62) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%65, %arg35, %arg36, %61, %60, %59) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%61, %arg37, %arg38, %58) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%58, %arg39, %arg40, %65, %57, %56, %55, %54) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%57, %arg41, %arg42, %53, %52, %51) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    byre.compute @ftv4.layernorm(%53, %arg43, %arg44, %50, %49, %48) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @AliasOp(%50, %47) {offset = 0 : i32} : memref<2x128x128xf32>, memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%47, %arg4, %46) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    %124 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%46, %arg45, %arg46, %46, %124) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32, 3 : i32, 2 : i32], kernel_name = "Unknown5_kernel"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceMaxOpf32f32(%124, %45) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %125 = memref.alloc() : memref<256x30522xf32>
    %126 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%124, %45, %125, %126) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6_kernel"} : memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%126, %44) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %127 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%44, %127) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7_kernel"} : memref<256xf32>, memref<256xf32>
    %128 = memref.alloc() : memref<256x30522xf32>
    %129 = memref.alloc() : memref<256x30522xf32>
    %130 = memref.alloc() : memref<256x30522xf32>
    %131 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%113, %112, %128, %125, %127, %129, %130, %131) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_ranks = [1 : i32, 1 : i32, 2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8_kernel"} : memref<256xi1>, memref<256xi64>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%128, %43) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%128, %42) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %132 = memref.alloc() : memref<f32>
    byre.compute @PTXOp(%42, %132) {BlockSize.x = 32 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown9_kernel"} : memref<f32>, memref<f32>
    %133 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%130, %132, %133) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_ranks = [2 : i32, 0 : i32, 2 : i32], kernel_name = "Unknown10_kernel"} : memref<256x30522xf32>, memref<f32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%133, %41) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%129, %40) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%40, %43, %arg47) {BlockSize.x = 32 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown11_kernel"} : memref<f32>, memref<f32>, memref<f32>
    %134 = memref.alloc() : memref<256x30522xf32>
    %135 = memref.alloc() : memref<2x128x30522xf32>
    byre.compute @PTXOp(%133, %131, %41, %134, %41, %135) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_ranks = [2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12_kernel"} : memref<256x30522xf32>, memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256xf32>, memref<2x128x30522xf32>
    %136 = memref.alloc() : memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%47, %134, %136) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%134, %arg4, %39) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    byre.compute @AliasOp(%39, %38) {offset = 0 : i32} : memref<256x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%38, %53, %arg43, %49, %48, %37, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%37, %57, %arg41, %52, %51, %36, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%36, %54, %arg39, %56, %55, %35, %arg83, %arg84, %34) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%35, %61, %arg37, %33, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%33, %65, %arg35, %60, %59, %32, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%34, %32, %137) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown13_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%137, %62, %arg33, %64, %63, %31, %arg77, %arg78, %30) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%31, %67, %arg31, %29, %arg75, %arg76) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%29, %28) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%28, %27) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%27, %73, %70, %26, %25) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%26, %73, %71, %24) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%24, %76, %75, %23, %22) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%23, %80, %arg25, %21, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%25, %80, %arg29, %20, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%22, %80, %arg27, %19, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %138 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%30, %21, %20, %19, %138) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%138, %77, %arg23, %79, %78, %18, %arg67, %arg68, %17) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%18, %84, %arg21, %16, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%16, %88, %arg19, %83, %82, %15, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %139 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%17, %15, %139) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%139, %85, %arg17, %87, %86, %14, %arg61, %arg62, %13) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%14, %90, %arg15, %12, %arg59, %arg60) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%12, %11) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%11, %10) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%10, %96, %93, %9, %8) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%9, %96, %94, %7) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%7, %99, %98, %6, %5) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%6, %102, %arg9, %4, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%8, %102, %arg13, %3, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%5, %102, %arg11, %2, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %140 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%13, %4, %3, %2, %140) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%140, %123, %arg7, %101, %100, %1, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %141 = memref.alloc() : memref<256x128xf32>
    %142 = memref.alloc() : memref<256x128xf32>
    byre.compute @PTXOp(%116, %1, %141, %119, %142) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_ranks = [2 : i32, 3 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown17_kernel"} : memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%136, %115, %141, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%110, %118, %142, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    byre.compute @ReduceSumOpf32f32(%1, %0) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %143 = memref.alloc() : memref<128x128xf32>
    byre.compute @PTXOp(%122, %0, %143) {BlockSize.x = 32 : i32, GridSize.x = 512 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18_kernel"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%111, %121, %143, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%135, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

