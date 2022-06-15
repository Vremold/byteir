// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  func @main(%arg0: memref<2x128xi64, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64, "cuda"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64, "cuda"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32, "cuda"> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32, "cuda"> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32, "cuda"> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32, "cuda"> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32, "cuda"> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32, "cuda"> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32, "cuda"> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32, "cuda"> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32, "cuda"> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32, "cuda"> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32, "cuda"> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32, "cuda"> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32, "cuda"> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32, "cuda"> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32, "cuda"> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32, "cuda"> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32, "cuda"> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32, "cuda"> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32, "cuda"> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32, "cuda"> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32, "cuda"> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32, "cuda"> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32, "cuda"> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32, "cuda"> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32, "cuda"> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32, "cuda"> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32, "cuda"> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32, "cuda"> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32, "cuda"> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32, "cuda"> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32, "cuda"> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32, "cuda"> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32, "cuda"> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32, "cuda"> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32, "cuda"> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32, "cuda"> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32, "cuda"> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32, "cuda"> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32, "cuda"> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32, "cuda"> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32, "cuda"> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32, "cuda"> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32, "cuda"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32, "cuda"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32, "cuda"> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32, "cuda"> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32, "cuda"> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32, "cuda"> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32, "cuda"> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32, "cuda"> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32, "cuda"> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32, "cuda"> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32, "cuda"> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32, "cuda"> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32, "cuda"> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32, "cuda"> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32, "cuda"> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32, "cuda"> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32, "cuda"> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32, "cuda"> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32, "cuda"> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32, "cuda"> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32, "cuda"> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32, "cuda"> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32, "cuda"> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32, "cuda"> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32, "cuda"> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32, "cuda"> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32, "cuda"> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32, "cuda"> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32, "cuda"> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32, "cuda"> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32, "cuda"> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32, "cuda"> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32, "cuda"> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32, "cuda"> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32, "cuda"> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32, "cuda"> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32, "cuda"> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32, "cuda"> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32, "cuda"> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32, "cuda"> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32, "cuda"> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32, "cuda"> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
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
    %55 = memref.alloc() : memref<31254528xi8, "cuda">
    %56 = memref.alloc() : memref<31254528xi8, "cuda">
    %57 = memref.alloc() : memref<512x128xf32, "cuda">
    byre.compute @FillOp(%57) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32, "cuda">
    %58 = memref.alloc() : memref<2x128xf32, "cuda">
    byre.compute @FillOp(%58) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32, "cuda">
    %59 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @FillOp(%59) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32, "cuda">
    %60 = memref.alloc() : memref<128xi64, "cuda">
    byre.compute @AliasOp(%arg2, %60) {arg_alias, offset = 0 : i32} : memref<1x512xi64, "cuda">, memref<128xi64, "cuda">
    %61 = memref.alloc() : memref<1x128xi64, "cuda">
    byre.compute @AliasOp(%arg3, %61) {arg_alias, offset = 0 : i32} : memref<1x512xi64, "cuda">, memref<1x128xi64, "cuda">
    %62 = memref.alloc() : memref<256xi64, "cuda">
    byre.compute @AliasOp(%arg1, %62) {arg_alias, offset = 0 : i32} : memref<2x128xi64, "cuda">, memref<256xi64, "cuda">
    %63 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%6, %63) {offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%arg1, %63) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0"} : memref<2x128xi64, "cuda">, memref<256xi1, "cuda">
    byre.compute @AliasOp(%arg1, %62) {arg_alias, offset = 0 : i32} : memref<2x128xi64, "cuda">, memref<256xi64, "cuda">
    %64 = memref.alloc() : memref<256xui32, "cuda">
    byre.compute @AliasOp(%56, %64) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256xui32, "cuda">
    %65 = memref.alloc() : memref<256x1xi64, "cuda">
    byre.compute @AliasOp(%21, %65) {offset = 0 : index} : memref<2048xi8, "cuda">, memref<256x1xi64, "cuda">
    %66 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%5, %66) {offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%arg0, %64, %65, %66) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1"} : memref<2x128xi64, "cuda">, memref<256xui32, "cuda">, memref<256x1xi64, "cuda">, memref<256xi1, "cuda">
    %67 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%55, %67) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %64, %67) {dim = 0 : i32} : memref<30522x128xf32, "cuda">, memref<256xui32, "cuda">, memref<256x128xf32, "cuda">
    %68 = memref.alloc() : memref<256xui32, "cuda">
    byre.compute @AliasOp(%55, %68) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<256xui32, "cuda">
    %69 = memref.alloc() : memref<256x1xi64, "cuda">
    byre.compute @AliasOp(%22, %69) {offset = 0 : index} : memref<2048xi8, "cuda">, memref<256x1xi64, "cuda">
    %70 = memref.alloc() : memref<256xi1, "cuda">
    byre.compute @AliasOp(%4, %70) {offset = 0 : index} : memref<32xi8, "cuda">, memref<256xi1, "cuda">
    byre.compute @PTXOp(%60, %68, %69, %70) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2"} : memref<128xi64, "cuda">, memref<256xui32, "cuda">, memref<256x1xi64, "cuda">, memref<256xi1, "cuda">
    %71 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%56, %71) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %68, %71) {dim = 0 : i32} : memref<2x128xf32, "cuda">, memref<256xui32, "cuda">, memref<256x128xf32, "cuda">
    %72 = memref.alloc() : memref<128xui32, "cuda">
    byre.compute @AliasOp(%55, %72) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<128xui32, "cuda">
    %73 = memref.alloc() : memref<128x1xi64, "cuda">
    byre.compute @AliasOp(%19, %73) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<128x1xi64, "cuda">
    %74 = memref.alloc() : memref<128xi1, "cuda">
    byre.compute @AliasOp(%3, %74) {offset = 0 : index} : memref<16xi8, "cuda">, memref<128xi1, "cuda">
    byre.compute @PTXOp(%61, %72, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3"} : memref<1x128xi64, "cuda">, memref<128xui32, "cuda">, memref<128x1xi64, "cuda">, memref<128xi1, "cuda">
    %75 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %75) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %72, %75) {dim = 0 : i32} : memref<512x128xf32, "cuda">, memref<128xui32, "cuda">, memref<128x128xf32, "cuda">
    %76 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%33, %76) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%67, %71, %75, %76) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4"} : memref<256x128xf32, "cuda">, memref<256x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %77 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%40, %77) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %78 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%18, %78) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %79 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%17, %79) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ftv4.layernorm(%76, %arg7, %arg8, %77, %78, %79) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %80 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%35, %80) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%77, %arg9, %arg10, %80) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %81 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%36, %81) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%77, %arg11, %arg12, %81) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %82 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %82) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.matmul(%80, %81, %82) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    %83 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%46, %83) {offset = 0 : index} : memref<262144xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %84 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %84) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %85 = memref.alloc() : memref<2x2x128x128xui8, "cuda">
    byre.compute @AliasOp(%23, %85) {offset = 0 : index} : memref<65536xi8, "cuda">, memref<2x2x128x128xui8, "cuda">
    byre.compute @ftv4.softmax(%82, %59, %83, %84, %85) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">
    %86 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%37, %86) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%77, %arg13, %arg14, %86) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %87 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%56, %87) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul(%83, %86, %87) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %88 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%38, %88) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    byre.compute @ftv4.transpose4d(%87, %88) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32, "cuda">, memref<2x128x2x64xf32, "cuda">
    %89 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%38, %89) {offset = 0 : i32} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %90 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %90) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%89, %arg15, %arg16, %90) : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %91 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%39, %91) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %92 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%16, %92) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %93 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%15, %93) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %94 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%25, %94) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%90, %arg17, %arg18, %77, %91, %92, %93, %94) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %95 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%51, %95) {offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %96 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%50, %96) {offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %97 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%2, %97) {offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%91, %arg19, %arg20, %95, %96, %97) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">
    %98 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %98) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%95, %arg21, %arg22, %98) : memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %99 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%26, %99) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %100 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%14, %100) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %101 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%13, %101) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %102 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%27, %102) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%98, %arg23, %arg24, %91, %99, %100, %101, %102) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %103 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%28, %103) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%99, %arg25, %arg26, %103) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %104 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%32, %104) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%99, %arg27, %arg28, %104) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %105 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %105) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.matmul(%103, %104, %105) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">
    %106 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%47, %106) {offset = 0 : index} : memref<262144xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %107 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %107) {offset = 262144 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %108 = memref.alloc() : memref<2x2x128x128xui8, "cuda">
    byre.compute @AliasOp(%24, %108) {offset = 0 : index} : memref<65536xi8, "cuda">, memref<2x2x128x128xui8, "cuda">
    byre.compute @ftv4.softmax(%105, %59, %106, %107, %108) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">
    %109 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%29, %109) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.linear_transpose(%99, %arg29, %arg30, %109) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %110 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%56, %110) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul(%106, %109, %110) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %111 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%30, %111) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    byre.compute @ftv4.transpose4d(%110, %111) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32, "cuda">, memref<2x128x2x64xf32, "cuda">
    %112 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%30, %112) {offset = 0 : i32} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %113 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %113) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%112, %arg31, %arg32, %113) : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %114 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%31, %114) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %115 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%12, %115) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %116 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%11, %116) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %117 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%34, %117) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%113, %arg33, %arg34, %99, %114, %115, %116, %117) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %118 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%49, %118) {offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %119 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%48, %119) {offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x512xf32, "cuda">
    %120 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%0, %120) {offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%114, %arg35, %arg36, %118, %119, %120) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">
    %121 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %121) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear(%118, %arg37, %arg38, %121) : memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %122 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%41, %122) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %123 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%10, %123) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %124 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%9, %124) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %125 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%42, %125) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_residual(%121, %arg39, %arg40, %114, %122, %123, %124, %125) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %126 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%43, %126) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %127 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%44, %127) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %128 = memref.alloc() : memref<0xf32, "cuda">
    byre.compute @AliasOp(%1, %128) {offset = 0 : index} : memref<0xi8, "cuda">, memref<0xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout(%122, %arg41, %arg42, %126, %127, %128) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<0xf32, "cuda">
    %129 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%45, %129) {offset = 0 : index} : memref<131072xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %130 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%20, %130) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    %131 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%8, %131) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ftv4.layernorm(%126, %arg43, %arg44, %129, %130, %131) : memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %132 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%45, %132) {offset = 0 : i32} : memref<131072xi8, "cuda">, memref<256x128xf32, "cuda">
    %133 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%56, %133) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%132, %arg4, %133) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32, "cuda">, memref<30522x128xf32, "cuda">, memref<256x30522xf32, "cuda">
    %134 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%55, %134) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%133, %arg45, %arg46, %134) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32, 2 : i32], kernel_name = "Unknown5"} : memref<256x30522xf32, "cuda">, memref<30522xf32, "cuda">, memref<2x128x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %135 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%53, %135) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceMaxOpf32f32(%134, %135) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %136 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%56, %136) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %137 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %137) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%135, %134, %136, %137) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %138 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%55, %138) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%137, %138) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %139 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%7, %139) {offset = 0 : index} : memref<1024xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @PTXOp(%138, %139) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7"} : memref<256xf32, "cuda">, memref<256xf32, "cuda">
    %140 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%52, %140) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %141 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%53, %141) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %142 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %142) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %143 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%55, %143) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%139, %136, %62, %63, %140, %141, %142, %143) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256xi64, "cuda">, memref<256xi1, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %144 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%arg1, %144) {arg_alias, offset = 0 : i32} : memref<2x128xi64, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%140, %144) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    %145 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%56, %145) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%140, %145) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    %146 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%52, %146) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%145, %146) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown9"} : memref<f32, "cuda">, memref<f32, "cuda">
    %147 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%56, %147) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    byre.compute @PTXOp(%146, %142, %147) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown10"} : memref<f32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">
    %148 = memref.alloc() : memref<256xf32, "cuda">
    byre.compute @AliasOp(%52, %148) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%147, %148) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32, "cuda">, memref<256xf32, "cuda">
    %149 = memref.alloc() : memref<f32, "cuda">
    byre.compute @AliasOp(%54, %149) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<f32, "cuda">
    byre.compute @ReduceSumOpf32f32(%141, %149) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32, "cuda">, memref<f32, "cuda">
    byre.compute @PTXOp(%149, %144, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown11"} : memref<f32, "cuda">, memref<f32, "cuda">, memref<f32, "cuda">
    %150 = memref.alloc() : memref<256x30522xf32, "cuda">
    byre.compute @AliasOp(%54, %150) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x30522xf32, "cuda">
    %151 = memref.alloc() : memref<2x128x30522xf32, "cuda">
    byre.compute @AliasOp(%53, %151) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x30522xf32, "cuda">
    byre.compute @PTXOp(%148, %143, %147, %150, %151) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12"} : memref<256xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<2x128x30522xf32, "cuda">
    %152 = memref.alloc() : memref<30522x128xf32, "cuda">
    byre.compute @AliasOp(%56, %152) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<30522x128xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%132, %150, %152) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32, "cuda">, memref<256x30522xf32, "cuda">, memref<30522x128xf32, "cuda">
    %153 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%56, %153) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @MatmulOpf32f32f32(%150, %arg4, %153) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32, "cuda">, memref<30522x128xf32, "cuda">, memref<256x128xf32, "cuda">
    %154 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %154) {offset = 15627264 : i32} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %155 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %155) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward(%154, %126, %arg43, %130, %131, %155, %arg87, %arg88) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %156 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %156) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%155, %122, %arg41, %127, %128, %156, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %157 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %157) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %158 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%52, %158) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%156, %125, %arg39, %123, %124, %157, %arg83, %arg84, %158) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %159 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%56, %159) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x512xf32, "cuda">
    byre.compute @ftv4.linear_backward(%157, %118, %arg37, %159, %arg81, %arg82) : memref<2x128x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">
    %160 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %160) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%159, %114, %arg35, %119, %120, %160, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">
    %161 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %161) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%158, %160, %161) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %162 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %162) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %163 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%52, %163) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%161, %117, %arg33, %115, %116, %162, %arg77, %arg78, %163) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %164 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %164) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_backward(%162, %112, %arg31, %164, %arg75, %arg76) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %165 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%55, %165) {offset = 0 : i32} : memref<31254528xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    %166 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%56, %166) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.transpose4d_backward(%165, %166) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %167 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %167) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %168 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %168) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%166, %106, %109, %167, %168) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %169 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %169) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.softmax_backward(%167, %106, %108, %169) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %170 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%55, %170) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    %171 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%52, %171) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%169, %103, %104, %170, %171) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %172 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%49, %172) {offset = 0 : index} : memref<524288xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%170, %99, %arg25, %172, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %173 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %173) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%168, %99, %arg29, %173, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %174 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %174) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%171, %99, %arg27, %174, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %175 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %175) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%163, %172, %173, %174, %175) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %176 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %176) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %177 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %177) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%175, %102, %arg23, %100, %101, %176, %arg67, %arg68, %177) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %178 = memref.alloc() : memref<2x128x512xf32, "cuda">
    byre.compute @AliasOp(%56, %178) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x512xf32, "cuda">
    byre.compute @ftv4.linear_backward(%176, %95, %arg21, %178, %arg65, %arg66) : memref<2x128x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<128x512xf32, "cuda">, memref<128xf32, "cuda">
    %179 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %179) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_gelu_dropout_backward(%178, %91, %arg19, %96, %97, %179, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<2x128x512xf32, "cuda">, memref<0xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<512x128xf32, "cuda">, memref<512xf32, "cuda">
    %180 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %180) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%177, %179, %180) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %181 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %181) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    %182 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %182) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward_residual(%180, %94, %arg17, %92, %93, %181, %arg61, %arg62, %182) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %183 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %183) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_backward(%181, %89, %arg15, %183, %arg59, %arg60) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %184 = memref.alloc() : memref<2x128x2x64xf32, "cuda">
    byre.compute @AliasOp(%56, %184) {offset = 15627264 : i32} : memref<31254528xi8, "cuda">, memref<2x128x2x64xf32, "cuda">
    %185 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%54, %185) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.transpose4d_backward(%184, %185) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %186 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %186) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %187 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%52, %187) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%185, %83, %86, %186, %187) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %188 = memref.alloc() : memref<2x2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %188) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x128xf32, "cuda">
    byre.compute @ftv4.softmax_backward(%186, %83, %85, %188) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xf32, "cuda">, memref<2x2x128x128xui8, "cuda">, memref<2x2x128x128xf32, "cuda">
    %189 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%55, %189) {offset = 131072 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    %190 = memref.alloc() : memref<2x2x128x64xf32, "cuda">
    byre.compute @AliasOp(%55, %190) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x2x128x64xf32, "cuda">
    byre.compute @ftv4.matmul_backward(%188, %80, %81, %189, %190) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">, memref<2x2x128x64xf32, "cuda">
    %191 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %191) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%189, %77, %arg9, %191, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %192 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %192) {offset = 15758336 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%187, %77, %arg13, %192, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %193 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %193) {offset = 15889408 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.linear_transpose_backward(%190, %77, %arg11, %193, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">, memref<128xf32, "cuda">
    %194 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %194) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @PTXOp(%182, %191, %192, %193, %194) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17"} : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">
    %195 = memref.alloc() : memref<2x128x128xf32, "cuda">
    byre.compute @AliasOp(%54, %195) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<2x128x128xf32, "cuda">
    byre.compute @ftv4.layernorm_backward(%194, %76, %arg7, %78, %79, %195, %arg51, %arg52) : memref<2x128x128xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<256xf32, "cuda">, memref<256xf32, "cuda">, memref<2x128x128xf32, "cuda">, memref<128xf32, "cuda">, memref<128xf32, "cuda">
    %196 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%56, %196) {offset = 15627264 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    %197 = memref.alloc() : memref<256x128xf32, "cuda">
    byre.compute @AliasOp(%55, %197) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @PTXOp(%66, %195, %70, %196, %197) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18"} : memref<256xi1, "cuda">, memref<2x128x128xf32, "cuda">, memref<256xi1, "cuda">, memref<256x128xf32, "cuda">, memref<256x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%152, %65, %196, %arg48) {dim = 0 : i32} : memref<30522x128xf32, "cuda">, memref<256x1xi64, "cuda">, memref<256x128xf32, "cuda">, memref<30522x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%58, %69, %197, %arg49) {dim = 0 : i32} : memref<2x128xf32, "cuda">, memref<256x1xi64, "cuda">, memref<256x128xf32, "cuda">, memref<2x128xf32, "cuda">
    %198 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%55, %198) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%195, %198) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32, "cuda">, memref<128x128xf32, "cuda">
    %199 = memref.alloc() : memref<128x128xf32, "cuda">
    byre.compute @AliasOp(%56, %199) {offset = 0 : index} : memref<31254528xi8, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @PTXOp(%74, %198, %199) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19"} : memref<128xi1, "cuda">, memref<128x128xf32, "cuda">, memref<128x128xf32, "cuda">
    byre.compute @IndexPutOpf32i64f32f32(%57, %73, %199, %arg50) {dim = 0 : i32} : memref<512x128xf32, "cuda">, memref<128x1xi64, "cuda">, memref<128x128xf32, "cuda">, memref<512x128xf32, "cuda">
    byre.compute @ReduceSumOpf32f32(%151, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32, "cuda">, memref<30522xf32, "cuda">
    return
  }
}

