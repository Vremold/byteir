// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main
module attributes {byre.container_module, gpu.container_module} {
  func.func @main(%arg0: memref<2x128xi64, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64, "cuda"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64, "cuda"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32, "cuda"> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32, "cuda"> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32, "cuda"> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32, "cuda"> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32, "cuda"> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32, "cuda"> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32, "cuda"> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32, "cuda"> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32, "cuda"> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32, "cuda"> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32, "cuda"> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32, "cuda"> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32, "cuda"> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32, "cuda"> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32, "cuda"> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32, "cuda"> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32, "cuda"> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32, "cuda"> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32, "cuda"> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32, "cuda"> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32, "cuda"> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32, "cuda"> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32, "cuda"> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32, "cuda"> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32, "cuda"> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32, "cuda"> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32, "cuda"> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32, "cuda"> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32, "cuda"> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32, "cuda"> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32, "cuda"> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32, "cuda"> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32, "cuda"> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32, "cuda"> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32, "cuda"> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32, "cuda"> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32, "cuda"> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32, "cuda"> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32, "cuda"> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32, "cuda"> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32, "cuda"> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32, "cuda"> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32, "cuda"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32, "cuda"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32, "cuda"> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32, "cuda"> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32, "cuda"> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32, "cuda"> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32, "cuda"> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32, "cuda"> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32, "cuda"> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32, "cuda"> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32, "cuda"> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32, "cuda"> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32, "cuda"> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32, "cuda"> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32, "cuda"> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32, "cuda"> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32, "cuda"> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32, "cuda"> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32, "cuda"> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32, "cuda"> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32, "cuda"> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32, "cuda"> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32, "cuda"> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32, "cuda"> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32, "cuda"> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32, "cuda"> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32, "cuda"> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32, "cuda"> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32, "cuda"> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32, "cuda"> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32, "cuda"> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32, "cuda"> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32, "cuda"> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32, "cuda"> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32, "cuda"> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32, "cuda"> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32, "cuda"> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32, "cuda"> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32, "cuda"> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32, "cuda"> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32, "cuda"> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32, "cuda"> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %0 = memref.alloc() : memref<0xi8, "cuda">
    %1 = memref.alloc() : memref<0xi8, "cuda">
    %2 = memref.alloc() : memref<0xi8, "cuda">
    %3 = memref.alloc() : memref<16xi8, "cuda">
    %4 = memref.alloc() : memref<32xi8, "cuda">
    %5 = memref.alloc() : memref<32xi8, "cuda">
    %6 = memref.alloc() : memref<32xi8, "cuda">
    %7 = memref.alloc() : memref<1024xi8, "cuda">
    %8 = memref.alloc() : memref<1024xi8, "cuda">
    %9 = memref.alloc() : memref<1024xi8, "cuda">
    %10 = memref.alloc() : memref<1024xi8, "cuda">
    %11 = memref.alloc() : memref<1024xi8, "cuda">
    %12 = memref.alloc() : memref<1024xi8, "cuda">
    %13 = memref.alloc() : memref<1024xi8, "cuda">
    %14 = memref.alloc() : memref<1024xi8, "cuda">
    %15 = memref.alloc() : memref<1024xi8, "cuda">
    %16 = memref.alloc() : memref<1024xi8, "cuda">
    %17 = memref.alloc() : memref<1024xi8, "cuda">
    %18 = memref.alloc() : memref<1024xi8, "cuda">
    %19 = memref.alloc() : memref<1024xi8, "cuda">
    %20 = memref.alloc() : memref<1024xi8, "cuda">
    %21 = memref.alloc() : memref<2048xi8, "cuda">
    %22 = memref.alloc() : memref<2048xi8, "cuda">
    %23 = memref.alloc() : memref<65536xi8, "cuda">
    %24 = memref.alloc() : memref<65536xi8, "cuda">
    %25 = memref.alloc() : memref<131072xi8, "cuda">
    %26 = memref.alloc() : memref<131072xi8, "cuda">
    %27 = memref.alloc() : memref<131072xi8, "cuda">
    %28 = memref.alloc() : memref<131072xi8, "cuda">
    %29 = memref.alloc() : memref<131072xi8, "cuda">
    %30 = memref.alloc() : memref<131072xi8, "cuda">
    %31 = memref.alloc() : memref<131072xi8, "cuda">
    %32 = memref.alloc() : memref<131072xi8, "cuda">
    %33 = memref.alloc() : memref<131072xi8, "cuda">
    %34 = memref.alloc() : memref<131072xi8, "cuda">
    %35 = memref.alloc() : memref<131072xi8, "cuda">
    %36 = memref.alloc() : memref<131072xi8, "cuda">
    %37 = memref.alloc() : memref<131072xi8, "cuda">
    %38 = memref.alloc() : memref<131072xi8, "cuda">
    %39 = memref.alloc() : memref<131072xi8, "cuda">
    %40 = memref.alloc() : memref<131072xi8, "cuda">
    %41 = memref.alloc() : memref<131072xi8, "cuda">
    %42 = memref.alloc() : memref<131072xi8, "cuda">
    %43 = memref.alloc() : memref<131072xi8, "cuda">
    %44 = memref.alloc() : memref<131072xi8, "cuda">
    %45 = memref.alloc() : memref<131072xi8, "cuda">
    %46 = memref.alloc() : memref<262144xi8, "cuda">
    %47 = memref.alloc() : memref<262144xi8, "cuda">
    %48 = memref.alloc() : memref<524288xi8, "cuda">
    %49 = memref.alloc() : memref<524288xi8, "cuda">
    %50 = memref.alloc() : memref<524288xi8, "cuda">
    %51 = memref.alloc() : memref<524288xi8, "cuda">
    %52 = memref.alloc() : memref<31254528xi8, "cuda">
    %53 = memref.alloc() : memref<31254528xi8, "cuda">
    %54 = memref.alloc() : memref<31254528xi8, "cuda">
    %55 = memref.alloc() : memref<512x128xf32, "cuda">
    byre.compute @FillOp(%55) {device = "cuda", value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32, "cuda">
    %56 = memref.alloc() : memref<2x128xf32, "cuda">
    byre.compute @FillOp(%56) {device = "cuda", value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32, "cuda">
    %57 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @FillOp(%57) {device = "cuda", value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32, "cuda">
    %58 = memref.alloc() : memref<128xi64, "cuda">
    byre.compute @AliasOp(%arg2, %58) {arg_alias, device = "cuda", offset = 0 : i32} : memref<1x512xi64, "cuda">, memref<128xi64, "cuda">
    %59 = memref.alloc() : memref<1x128xi64, "cuda">
    byre.compute @AliasOp(%arg3, %59) {arg_alias, device = "cuda", offset = 0 : i32} : memref<1x512xi64, "cuda">, memref<1x128xi64, "cuda">
    %60 = memref.alloc() : memref<256xi64, "cuda">
    byre.compute @AliasOp(%arg1, %60) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128xi64, "cuda">, memref<256xi64, "cuda">
    %61 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%6, %61) {device = "cuda", offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%arg1, %61) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], device = "cuda", kernel_name = "Unknown0"} : memref<2x128xi64, "cuda">, memref<256xi1, "cuda">
    byre.compute @AliasOp(%arg1, %60) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128xi64, "cuda">, memref<256xi64, "cuda">
    %62 = memref.alloc() : memref<256xui32, "cuda">
    byre.compute @AliasOp(%54, %62) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<256xui32, "cuda">
    %63 = memref.alloc() : memref<256x1xi64, "cuda">
    byre.compute @AliasOp(%21, %63) {device = "cuda", offset = 0 : index} : memref<2048xi8, "cuda">, memref<256x1xi64, "cuda">
    %64 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%4, %64) {device = "cuda", offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%arg0, %62, %63, %64) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], device = "cuda", kernel_name = "Unknown1"} : memref<2x128xi64, "cuda">, memref<256xui32, "cuda">, memref<256x1xi64, "cuda">, memref<256xi1, "cuda">
    %65 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%54, %65) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %62, %65) {device = "cuda", dim = 0 : i32} : memref<30522x128xf32, "cuda">, memref<256xui32, "cuda">, memref<256x128xf32, "cuda">
    %66 = memref.alloc() : memref<256xui32, "cuda">
    byre.compute @AliasOp(%54, %66) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<256xui32, "cuda">
    %67 = memref.alloc() : memref<256x1xi64, "cuda">
    byre.compute @AliasOp(%22, %67) {device = "cuda", offset = 0 : index} : memref<2048xi8, "cuda">, memref<256x1xi64, "cuda">
    %68 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%5, %68) {device = "cuda", offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%58, %66, %67, %68) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown2"} : memref<128xi64, "cuda">, memref<256xui32, "cuda">, memref<256x1xi64, "cuda">, memref<256xi1, "cuda">
    %69 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%54, %69) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %66, %69) {device = "cuda", dim = 0 : i32} : memref<2x128xf32, "cuda">, memref<256xui32, "cuda">, memref<256x128xf32, "cuda">
    %70 = memref.alloc() : memref<128xui32, "cuda">
    byre.compute @AliasOp(%54, %70) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<128xui32, "cuda">
    %71 = memref.alloc() : memref<128x1xi64, "cuda">
    byre.compute @AliasOp(%14, %71) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<128x1xi64, "cuda">
    %72 = memref.alloc() : memref<128xi1, "cuda">
    byre.compute @AliasOp(%3, %72) {device = "cuda", offset = 0 : index} : memref<16xi8, "cuda">, memref<128xi1, "cuda">
    byre.compute @PTXOp(%59, %70, %71, %72) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], device = "cuda", kernel_name = "Unknown3"} : memref<1x128xi64, "cuda">, memref<128xui32, "cuda">, memref<128x1xi64, "cuda">, memref<128xi1, "cuda">
    %73 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %73) {device = "cuda", offset = 131072 : i32} : memref<31254528xi8, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %70, %73) {device = "cuda", dim = 0 : i32} : memref<512x128xf32, "cuda">, memref<128xui32, "cuda">, memref<128x128xf32, "cuda">
    %74 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%33, %74) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%65, %69, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], device = "cuda", kernel_name = "Unknown4"} : memref<256x128xf32, "cuda">, memref<256x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %75 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%40, %75) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %76 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%18, %76) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %77 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%17, %77) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ftv4.layernorm(%74, %arg7, %arg8, %75, %76, %77) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %78 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%35, %78) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%75, %arg9, %arg10, %78) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %79 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%36, %79) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%75, %arg11, %arg12, %79) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %80 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %80) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.matmul(%78, %79, %80) {device = "cuda", scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    %81 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%46, %81) {device = "cuda", offset = 0 : index} : memref<262144xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %82 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %82) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %83 = memref.alloc() : memref<2x2x128x128xui8, "cuda">
    byre.compute @AliasOp(%23, %83) {device = "cuda", offset = 0 : index} : memref<65536xi8, "cuda">, memref<2x2x128x128xui8, "cuda">
    byre.compute @ftv4.softmax(%80, %57, %81, %82, %83) {batch_first = true, device = "cuda", dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">
    %84 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%37, %84) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%75, %arg13, %arg14, %84) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %85 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %85) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul(%81, %84, %85) {device = "cuda", scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %86 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%38, %86) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    byre.compute @ftv4.transpose4d(%85, %86) {device = "cuda", forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32, "cuda">, memref<2x128x2x64xf32, "cuda">
    %87 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%38, %87) {device = "cuda", offset = 0 : i32} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %88 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %88) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%87, %arg15, %arg16, %88) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %89 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%39, %89) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %90 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%16, %90) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %91 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%15, %91) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %92 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%25, %92) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%88, %arg17, %arg18, %75, %89, %90, %91, %92) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %93 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%49, %93) {device = "cuda", offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %94 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%50, %94) {device = "cuda", offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %95 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%2, %95) {device = "cuda", offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%89, %arg19, %arg20, %93, %94, %95) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">
    %96 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %96) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%93, %arg21, %arg22, %96) {device = "cuda"} : memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %97 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%26, %97) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %98 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%13, %98) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %99 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%12, %99) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %100 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%27, %100) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%96, %arg23, %arg24, %89, %97, %98, %99, %100) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %101 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%28, %101) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%97, %arg25, %arg26, %101) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %102 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%32, %102) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%97, %arg27, %arg28, %102) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %103 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %103) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.matmul(%101, %102, %103) {device = "cuda", scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    %104 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%47, %104) {device = "cuda", offset = 0 : index} : memref<262144xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %105 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %105) {device = "cuda", offset = 262144 : i32} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %106 = memref.alloc() : memref<2x2x128x128xui8, "cuda">
    byre.compute @AliasOp(%24, %106) {device = "cuda", offset = 0 : index} : memref<65536xi8, "cuda">, memref<2x2x128x128xui8, "cuda">
    byre.compute @ftv4.softmax(%103, %57, %104, %105, %106) {batch_first = true, device = "cuda", dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">
    %107 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%29, %107) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%97, %arg29, %arg30, %107) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %108 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %108) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul(%104, %107, %108) {device = "cuda", scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %109 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%30, %109) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    byre.compute @ftv4.transpose4d(%108, %109) {device = "cuda", forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32, "cuda">, memref<2x128x2x64xf32, "cuda">
    %110 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%30, %110) {device = "cuda", offset = 0 : i32} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %111 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %111) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%110, %arg31, %arg32, %111) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %112 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%31, %112) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %113 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%11, %113) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %114 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%10, %114) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %115 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%34, %115) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%111, %arg33, %arg34, %97, %112, %113, %114, %115) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %116 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%48, %116) {device = "cuda", offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %117 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%51, %117) {device = "cuda", offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %118 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%1, %118) {device = "cuda", offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%112, %arg35, %arg36, %116, %117, %118) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">
    %119 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %119) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%116, %arg37, %arg38, %119) {device = "cuda"} : memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %120 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%43, %120) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %121 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%19, %121) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %122 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%9, %122) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %123 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%44, %123) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%119, %arg39, %arg40, %112, %120, %121, %122, %123) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %124 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%42, %124) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %125 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%45, %125) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %126 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%0, %126) {device = "cuda", offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%120, %arg41, %arg42, %124, %125, %126) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<0xf32, "cuda">
    %127 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%41, %127) {device = "cuda", offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %128 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%20, %128) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %129 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%8, %129) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ftv4.layernorm(%124, %arg43, %arg44, %127, %128, %129) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %130 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%41, %130) {device = "cuda", offset = 0 : i32} : memref<131072xi8, "cuda">, memref<256x128xf32, "cuda">
    %131 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %131) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%130, %arg4, %131) {device = "cuda", lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32, "cuda">, memref<30522x128xf32, "cuda">, memref<256x30522xf32, "cuda">
    %132 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%arg46, %132) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%131, %arg45, %arg46) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown5"} : memref<256x30522xf32, "cuda">, memref<30522xf32, "cuda">, memref<2x128x30522xf32, "cuda">
    byre.compute @AliasOp(%arg46, %132) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %133 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%53, %133) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceMaxOpf32f32(%132, %133) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %134 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %134) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %135 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %135) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%133, %132, %134, %135) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown6"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %136 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%arg46, %136) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%135, %136) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %137 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%7, %137) {device = "cuda", offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%136, %137) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], device = "cuda", kernel_name = "Unknown7"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %138 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %138) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %139 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%arg46, %139) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %140 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%53, %140) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %141 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%52, %141) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%137, %134, %60, %61, %138, %139, %140, %141) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown8"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256xi64, "cuda">, memref<256xi1, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %142 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%54, %142) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%139, %142) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    %143 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%54, %143) {device = "cuda", offset = 4 : i32} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%138, %143) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%142, %143, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], device = "cuda", kernel_name = "Unknown9"} : memref<f32, "cuda">, memref<f32, "cuda">, memref<f32, "cuda">
    %144 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%54, %144) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%138, %144) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    %145 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%54, %145) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%144, %145) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], device = "cuda", kernel_name = "Unknown10"} : memref<f32, "cuda">, memref<f32, "cuda">
    %146 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %146) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%145, %140, %146) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown11"} : memref<f32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %147 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%arg46, %147) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%146, %147) {device = "cuda", dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %148 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %148) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %149 = memref.alloc() : memref<2x128x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %149) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x30522xf32, "cuda">
    byre.compute @PTXOp(%147, %141, %146, %148) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown12"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %149) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x30522xf32, "cuda">
    %150 = memref.alloc() : memref<30522x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %150) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<30522x128xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%130, %148, %150) {device = "cuda", lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<30522x128xf32, "cuda">
    %151 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%53, %151) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%148, %arg4, %151) {device = "cuda", lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32, "cuda">, memref<30522x128xf32, "cuda">, memref<256x128xf32, "cuda">
    %152 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%53, %152) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %153 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %153) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward(%152, %124, %arg43, %128, %129, %153, %arg87, %arg88) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %154 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %154) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%153, %120, %arg41, %125, %126, %154, %arg85, %arg86) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %155 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %155) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %156 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%53, %156) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%154, %123, %arg39, %121, %122, %155, %arg83, %arg84, %156) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %157 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%54, %157) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x512xf32, "cuda">
    byre.compute @ftv4.linear_backward(%155, %116, %arg37, %157, %arg81, %arg82) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">
    %158 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %158) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%157, %112, %arg35, %117, %118, %158, %arg79, %arg80) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">
    %159 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %159) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%156, %158, %159) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], device = "cuda", kernel_name = "Unknown14"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %160 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %160) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %161 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%52, %161) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%159, %115, %arg33, %113, %114, %160, %arg77, %arg78, %161) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %162 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %162) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_backward(%160, %110, %arg31, %162, %arg75, %arg76) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %163 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%54, %163) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    %164 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%arg46, %164) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.transpose4d_backward(%163, %164) {device = "cuda", forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %165 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %165) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %166 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%53, %166) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%164, %104, %107, %165, %166) {device = "cuda", scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %167 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %167) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.softmax_backward(%165, %104, %106, %167) {device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %168 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %168) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    %169 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%52, %169) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%167, %101, %102, %168, %169) {device = "cuda", scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %170 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %170) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%168, %97, %arg25, %170, %arg69, %arg70) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %171 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %171) {arg_alias, device = "cuda", offset = 15758336 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%166, %97, %arg29, %171, %arg73, %arg74) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %172 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %172) {arg_alias, device = "cuda", offset = 15889408 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%169, %97, %arg27, %172, %arg71, %arg72) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %173 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %173) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%161, %170, %171, %172, %173) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], device = "cuda", kernel_name = "Unknown15"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %174 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %174) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %175 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%53, %175) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%173, %100, %arg23, %98, %99, %174, %arg67, %arg68, %175) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %176 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%54, %176) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x512xf32, "cuda">
    byre.compute @ftv4.linear_backward(%174, %93, %arg21, %176, %arg65, %arg66) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">
    %177 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %177) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%176, %89, %arg19, %94, %95, %177, %arg63, %arg64) {act_gelu = true, device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">
    %178 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %178) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%175, %177, %178) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], device = "cuda", kernel_name = "Unknown16"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %179 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %179) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %180 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%53, %180) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%178, %92, %arg17, %90, %91, %179, %arg61, %arg62, %180) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %181 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %181) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_backward(%179, %87, %arg15, %181, %arg59, %arg60) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %182 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%54, %182) {device = "cuda", offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    %183 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%arg46, %183) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.transpose4d_backward(%182, %183) {device = "cuda", forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %184 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %184) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %185 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%53, %185) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%183, %81, %84, %184, %185) {device = "cuda", scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %186 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %186) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.softmax_backward(%184, %81, %83, %186) {device = "cuda", dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %187 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %187) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    %188 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %188) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%186, %78, %79, %187, %188) {device = "cuda", scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %189 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %189) {arg_alias, device = "cuda", offset = 15627264 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%187, %75, %arg9, %189, %arg53, %arg54) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %190 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %190) {arg_alias, device = "cuda", offset = 15758336 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%185, %75, %arg13, %190, %arg57, %arg58) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %191 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %191) {arg_alias, device = "cuda", offset = 15889408 : i32} : memref<2x128x30522xf32, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%188, %75, %arg11, %191, %arg55, %arg56) {device = "cuda", forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %192 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %192) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%180, %189, %190, %191, %192) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], device = "cuda", kernel_name = "Unknown17"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %193 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%53, %193) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward(%192, %74, %arg7, %76, %77, %193, %arg51, %arg52) {device = "cuda"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %194 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%54, %194) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    %195 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%53, %195) {device = "cuda", offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @PTXOp(%64, %193, %68, %194, %195) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown18"} : memref<256xi1, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xi1, "cuda">, memref<256x128xf32, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%150, %63, %194, %arg48) {device = "cuda", dim = 0 : i32} : memref<30522x128xf32, "cuda">, memref<256x1xi64, "cuda">, memref<256x128xf32, "cuda">, memref<30522x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%56, %67, %195, %arg49) {device = "cuda", dim = 0 : i32} : memref<2x128xf32, "cuda">, memref<256x1xi64, "cuda">, memref<256x128xf32, "cuda">, memref<2x128xf32, "cuda">
    %196 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%arg46, %196) {arg_alias, device = "cuda", offset = 0 : i32} : memref<2x128x30522xf32, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%193, %196) {device = "cuda", dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">
    %197 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %197) {device = "cuda", offset = 0 : index} : memref<31254528xi8, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @PTXOp(%72, %196, %197) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], device = "cuda", kernel_name = "Unknown19"} : memref<128xi1, "cuda">, memref<128x128xf32, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%55, %71, %197, %arg50) {device = "cuda", dim = 0 : i32} : memref<512x128xf32, "cuda">, memref<128x1xi64, "cuda">, memref<128x128xf32, "cuda">, memref<512x128xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%149, %arg89) {device = "cuda", dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32, "cuda">, memref<30522xf32, "cuda">
    return
  }
}

