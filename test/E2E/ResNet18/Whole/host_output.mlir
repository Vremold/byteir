// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
module @IrToMhlo.2452 attributes {byre.container_module, gpu.container_module} {
  func @main(%arg0: memref<4x3x224x224xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<4x1000xf32> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64x3x7x7xf32> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<64xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<64xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64x64x3x3xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<64xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<64xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<64x64x3x3xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<64xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<64xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<64xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<64xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<64x64x3x3xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<64xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<64xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<64xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<64xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<64x64x3x3xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<64xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<64xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<64xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<64xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x64x3x3xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128x128x3x3xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<128xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x64x1x1xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128x128x3x3xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<128xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<128xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<128x128x3x3xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<128xf32> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<128xf32> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<128xf32> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<128xf32> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<256x128x3x3xf32> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<256xf32> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<256xf32> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<256xf32> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<256xf32> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<256x256x3x3xf32> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<256xf32> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<256xf32> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<256xf32> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<256xf32> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<256x128x1x1xf32> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<256xf32> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<256xf32> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<256xf32> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<256xf32> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<256x256x3x3xf32> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<256xf32> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<256xf32> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<256xf32> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<256xf32> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<256x256x3x3xf32> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<256xf32> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<256xf32> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<256xf32> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<256xf32> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<512x256x3x3xf32> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<512xf32> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<512xf32> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<512xf32> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<512xf32> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<512x512x3x3xf32> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<512xf32> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<512xf32> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<512xf32> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<512xf32> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<512x256x1x1xf32> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<512xf32> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<512xf32> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<512xf32> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<512xf32> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<512x512x3x3xf32> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<512xf32> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<512xf32> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<512xf32> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<512xf32> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<512x512x3x3xf32> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<512xf32> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<512xf32> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<512xf32> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<512xf32> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<1000x512xf32> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<1000xf32> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<f32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg105: memref<64x3x7x7xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg106: memref<64xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg107: memref<64xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg108: memref<64x64x3x3xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg109: memref<64xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg110: memref<64xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg111: memref<64x64x3x3xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg112: memref<64xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg113: memref<64xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg114: memref<64x64x3x3xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg115: memref<64xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg116: memref<64xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg117: memref<64x64x3x3xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg118: memref<64xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg119: memref<64xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg120: memref<128x64x3x3xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg121: memref<128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg122: memref<128xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg123: memref<128x128x3x3xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg124: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg125: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg126: memref<128x64x1x1xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg127: memref<128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg128: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg129: memref<128x128x3x3xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg130: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg131: memref<128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg132: memref<128x128x3x3xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg133: memref<128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg134: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg135: memref<256x128x3x3xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg136: memref<256xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg137: memref<256xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg138: memref<256x256x3x3xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg139: memref<256xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg140: memref<256xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg141: memref<256x128x1x1xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg142: memref<256xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg143: memref<256xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg144: memref<256x256x3x3xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg145: memref<256xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg146: memref<256xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg147: memref<256x256x3x3xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg148: memref<256xf32> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg149: memref<256xf32> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg150: memref<512x256x3x3xf32> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg151: memref<512xf32> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg152: memref<512xf32> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg153: memref<512x512x3x3xf32> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg154: memref<512xf32> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg155: memref<512xf32> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg156: memref<512x256x1x1xf32> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg157: memref<512xf32> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg158: memref<512xf32> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg159: memref<512x512x3x3xf32> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg160: memref<512xf32> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg161: memref<512xf32> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg162: memref<512x512x3x3xf32> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg163: memref<512xf32> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg164: memref<512xf32> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg165: memref<1000x512xf32> {byre.argname = "Output61", byre.argtype = 2 : i32}, %arg166: memref<1000xf32> {byre.argname = "Output62", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %0 = memref.alloc() : memref<8xi8>
    %1 = memref.alloc() : memref<4096xi8>
    %2 = memref.alloc() : memref<8000xi8>
    %3 = memref.alloc() : memref<8000xi8>
    %4 = memref.alloc() : memref<16000xi8>
    %5 = memref.alloc() : memref<16000xi8>
    %6 = memref.alloc() : memref<16384xi8>
    %7 = memref.alloc() : memref<25088xi8>
    %8 = memref.alloc() : memref<25088xi8>
    %9 = memref.alloc() : memref<25088xi8>
    %10 = memref.alloc() : memref<50176xi8>
    %11 = memref.alloc() : memref<50176xi8>
    %12 = memref.alloc() : memref<50176xi8>
    %13 = memref.alloc() : memref<50176xi8>
    %14 = memref.alloc() : memref<65536xi8>
    %15 = memref.alloc() : memref<73728xi8>
    %16 = memref.alloc() : memref<73728xi8>
    %17 = memref.alloc() : memref<73728xi8>
    %18 = memref.alloc() : memref<73728xi8>
    %19 = memref.alloc() : memref<100352xi8>
    %20 = memref.alloc() : memref<100352xi8>
    %21 = memref.alloc() : memref<100352xi8>
    %22 = memref.alloc() : memref<100352xi8>
    %23 = memref.alloc() : memref<147456xi8>
    %24 = memref.alloc() : memref<200704xi8>
    %25 = memref.alloc() : memref<200704xi8>
    %26 = memref.alloc() : memref<200704xi8>
    %27 = memref.alloc() : memref<262144xi8>
    %28 = memref.alloc() : memref<294912xi8>
    %29 = memref.alloc() : memref<294912xi8>
    %30 = memref.alloc() : memref<294912xi8>
    %31 = memref.alloc() : memref<401408xi8>
    %32 = memref.alloc() : memref<401408xi8>
    %33 = memref.alloc() : memref<401408xi8>
    %34 = memref.alloc() : memref<401408xi8>
    %35 = memref.alloc() : memref<401408xi8>
    %36 = memref.alloc() : memref<401408xi8>
    %37 = memref.alloc() : memref<401408xi8>
    %38 = memref.alloc() : memref<589824xi8>
    %39 = memref.alloc() : memref<802816xi8>
    %40 = memref.alloc() : memref<802816xi8>
    %41 = memref.alloc() : memref<802816xi8>
    %42 = memref.alloc() : memref<802816xi8>
    %43 = memref.alloc() : memref<802816xi8>
    %44 = memref.alloc() : memref<802816xi8>
    %45 = memref.alloc() : memref<802816xi8>
    %46 = memref.alloc() : memref<802816xi8>
    %47 = memref.alloc() : memref<802816xi8>
    %48 = memref.alloc() : memref<1179648xi8>
    %49 = memref.alloc() : memref<1179648xi8>
    %50 = memref.alloc() : memref<1179648xi8>
    %51 = memref.alloc() : memref<1204224xi8>
    %52 = memref.alloc() : memref<1605632xi8>
    %53 = memref.alloc() : memref<1605632xi8>
    %54 = memref.alloc() : memref<1605632xi8>
    %55 = memref.alloc() : memref<1605632xi8>
    %56 = memref.alloc() : memref<1605632xi8>
    %57 = memref.alloc() : memref<1605632xi8>
    %58 = memref.alloc() : memref<1605632xi8>
    %59 = memref.alloc() : memref<1605632xi8>
    %60 = memref.alloc() : memref<2359296xi8>
    %61 = memref.alloc() : memref<4718592xi8>
    %62 = memref.alloc() : memref<4718592xi8>
    %63 = memref.alloc() : memref<4718592xi8>
    %64 = memref.alloc() : memref<6422528xi8>
    %65 = memref.alloc() : memref<6422528xi8>
    %66 = memref.alloc() : memref<6422528xi8>
    %67 = memref.alloc() : memref<4x3x224x224xf16>
    byre.compute @AliasOp(%51, %67) {offset = 0 : index} : memref<1204224xi8>, memref<4x3x224x224xf16>
    byre.compute @PTXOp(%arg0, %67) {BlockSize.x = 128 : i32, GridSize.x = 4704 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown0"} : memref<4x3x224x224xf32>, memref<4x3x224x224xf16>
    %68 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @AliasOp(%66, %68) {offset = 0 : index} : memref<6422528xi8>, memref<64x3x7x7xf16>
    byre.compute @PTXOp(%arg2, %68) {BlockSize.x = 128 : i32, GridSize.x = 74 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown1"} : memref<64x3x7x7xf32>, memref<64x3x7x7xf16>
    %69 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%65, %69) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    byre.compute @ConvOpf16f16f16(%67, %68, %69) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>
    %70 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%64, %70) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%69, %arg3, %arg4, %70) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf16>
    %71 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%17, %71) {offset = 0 : index} : memref<73728xi8>, memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg7, %71) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %72 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%18, %72) {offset = 0 : index} : memref<73728xi8>, memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg12, %72) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %73 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%16, %73) {offset = 0 : index} : memref<73728xi8>, memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg17, %73) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown5"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %74 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%15, %74) {offset = 0 : index} : memref<73728xi8>, memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg22, %74) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown6"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %75 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @AliasOp(%6, %75) {offset = 0 : index} : memref<16384xi8>, memref<128x64x1x1xf16>
    byre.compute @PTXOp(%arg37, %75) {BlockSize.x = 128 : i32, GridSize.x = 64 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown7"} : memref<128x64x1x1xf32>, memref<128x64x1x1xf16>
    %76 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @AliasOp(%23, %76) {offset = 0 : index} : memref<147456xi8>, memref<128x64x3x3xf16>
    byre.compute @PTXOp(%arg27, %76) {BlockSize.x = 128 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown8"} : memref<128x64x3x3xf32>, memref<128x64x3x3xf16>
    %77 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%28, %77) {offset = 0 : index} : memref<294912xi8>, memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg32, %77) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown9"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %78 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%29, %78) {offset = 0 : index} : memref<294912xi8>, memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg42, %78) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown10"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %79 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%30, %79) {offset = 0 : index} : memref<294912xi8>, memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg47, %79) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown11"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %80 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @AliasOp(%14, %80) {offset = 0 : index} : memref<65536xi8>, memref<256x128x1x1xf16>
    byre.compute @PTXOp(%arg62, %80) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown12"} : memref<256x128x1x1xf32>, memref<256x128x1x1xf16>
    %81 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @AliasOp(%42, %81) {offset = 0 : index} : memref<802816xi8>, memref<256x128x3x3xf16>
    byre.compute @PTXOp(%arg52, %81) {BlockSize.x = 128 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown13"} : memref<256x128x3x3xf32>, memref<256x128x3x3xf16>
    %82 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%53, %82) {offset = 0 : index} : memref<1605632xi8>, memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg57, %82) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown14"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %83 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%52, %83) {offset = 0 : index} : memref<1605632xi8>, memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg67, %83) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown15"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %84 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%50, %84) {offset = 0 : index} : memref<1179648xi8>, memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg72, %84) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown16"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %85 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @AliasOp(%27, %85) {offset = 0 : index} : memref<262144xi8>, memref<512x256x1x1xf16>
    byre.compute @PTXOp(%arg87, %85) {BlockSize.x = 128 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown17"} : memref<512x256x1x1xf32>, memref<512x256x1x1xf16>
    %86 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @AliasOp(%60, %86) {offset = 0 : index} : memref<2359296xi8>, memref<512x256x3x3xf16>
    byre.compute @PTXOp(%arg77, %86) {BlockSize.x = 128 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown18"} : memref<512x256x3x3xf32>, memref<512x256x3x3xf16>
    %87 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%62, %87) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg82, %87) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown19"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %88 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%61, %88) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg92, %88) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown20"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %89 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%63, %89) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg97, %89) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %90 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%2, %90) {offset = 0 : index} : memref<8000xi8>, memref<4x1000xf16>
    byre.compute @PTXOp(%arg1, %90) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown22"} : memref<4x1000xf32>, memref<4x1000xf16>
    %91 = memref.alloc() : memref<1000x512xf16>
    byre.compute @AliasOp(%49, %91) {offset = 0 : index} : memref<1179648xi8>, memref<1000x512xf16>
    byre.compute @PTXOp(%arg102, %91) {BlockSize.x = 128 : i32, GridSize.x = 4000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown23"} : memref<1000x512xf32>, memref<1000x512xf16>
    %92 = memref.alloc() : memref<1000xf16>
    byre.compute @AliasOp(%49, %92) {offset = 1024000 : index} : memref<1179648xi8>, memref<1000xf16>
    byre.compute @PTXOp(%arg103, %92) {BlockSize.x = 128 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown24"} : memref<1000xf32>, memref<1000xf16>
    %93 = memref.alloc() : memref<4xf16>
    byre.compute @AliasOp(%0, %93) {offset = 0 : index} : memref<8xi8>, memref<4xf16>
    byre.compute @ReduceSumOpf16f16(%90, %93) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %94 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%66, %94) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    %95 = memref.alloc() : memref<4x64x112x112xi1>
    byre.compute @AliasOp(%37, %95) {offset = 0 : index} : memref<401408xi8>, memref<4x64x112x112xi1>
    byre.compute @PTXOp(%70, %94, %95) {BlockSize.x = 128 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown25"} : memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
    %96 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %96) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @PoolMaxOpf16f16(%94, %96) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>
    %97 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%59, %97) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvOpf16f16f16(%96, %71, %97) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %98 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %98) {offset = 1605632 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%97, %arg8, %arg9, %98) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %99 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%58, %99) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    %100 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @AliasOp(%22, %100) {offset = 0 : index} : memref<100352xi8>, memref<4x64x56x56xi1>
    byre.compute @PTXOp(%98, %99, %100) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown27"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    %101 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %101) {offset = 1605632 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvOpf16f16f16(%99, %72, %101) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %102 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %102) {offset = 3211264 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%101, %arg13, %arg14, %102) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %103 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%57, %103) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    %104 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @AliasOp(%19, %104) {offset = 0 : index} : memref<100352xi8>, memref<4x64x56x56xi1>
    byre.compute @PTXOp(%102, %96, %103, %104) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown29"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    %105 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%56, %105) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvOpf16f16f16(%103, %73, %105) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %106 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %106) {offset = 3211264 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%105, %arg18, %arg19, %106) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %107 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%55, %107) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    %108 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @AliasOp(%21, %108) {offset = 0 : index} : memref<100352xi8>, memref<4x64x56x56xi1>
    byre.compute @PTXOp(%106, %107, %108) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown31"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    %109 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %109) {offset = 3211264 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvOpf16f16f16(%107, %74, %109) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %110 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%54, %110) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%109, %arg23, %arg24, %110) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %111 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %111) {offset = 4816896 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    %112 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @AliasOp(%20, %112) {offset = 0 : index} : memref<100352xi8>, memref<4x64x56x56xi1>
    byre.compute @PTXOp(%110, %103, %111, %112) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown33"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    %113 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%39, %113) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%111, %75, %113) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>
    %114 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%48, %114) {offset = 0 : index} : memref<1179648xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%113, %arg38, %arg39, %114) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %115 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%41, %115) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%111, %76, %115) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>
    %116 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %116) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%115, %arg28, %arg29, %116) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %117 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%43, %117) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    %118 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @AliasOp(%13, %118) {offset = 0 : index} : memref<50176xi8>, memref<4x128x28x28xi1>
    byre.compute @PTXOp(%116, %117, %118) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown36"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    %119 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%44, %119) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%117, %77, %119) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %120 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %120) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%119, %arg33, %arg34, %120) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %121 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%46, %121) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    %122 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @AliasOp(%12, %122) {offset = 0 : index} : memref<50176xi8>, memref<4x128x28x28xi1>
    byre.compute @PTXOp(%120, %114, %121, %122) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown38"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    %123 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%47, %123) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%121, %78, %123) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %124 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %124) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%123, %arg43, %arg44, %124) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %125 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%45, %125) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    %126 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @AliasOp(%11, %126) {offset = 0 : index} : memref<50176xi8>, memref<4x128x28x28xi1>
    byre.compute @PTXOp(%124, %125, %126) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown40"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    %127 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%40, %127) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%125, %79, %127) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %128 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%48, %128) {offset = 0 : index} : memref<1179648xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%127, %arg48, %arg49, %128) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %129 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %129) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    %130 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @AliasOp(%10, %130) {offset = 0 : index} : memref<50176xi8>, memref<4x128x28x28xi1>
    byre.compute @PTXOp(%128, %121, %129, %130) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown42"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    %131 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%36, %131) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%129, %80, %131) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>
    %132 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %132) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%131, %arg63, %arg64, %132) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %133 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%38, %133) {offset = 0 : index} : memref<589824xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%129, %81, %133) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>
    %134 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %134) {offset = 1204224 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%133, %arg53, %arg54, %134) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %135 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%35, %135) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    %136 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @AliasOp(%9, %136) {offset = 0 : index} : memref<25088xi8>, memref<4x256x14x14xi1>
    byre.compute @PTXOp(%134, %135, %136) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown45"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    %137 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%34, %137) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%135, %82, %137) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %138 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %138) {offset = 1204224 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%137, %arg58, %arg59, %138) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %139 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%33, %139) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    %140 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @AliasOp(%8, %140) {offset = 0 : index} : memref<25088xi8>, memref<4x256x14x14xi1>
    byre.compute @PTXOp(%138, %132, %139, %140) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown47"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    %141 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%52, %141) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%139, %83, %141) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %142 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %142) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%141, %arg68, %arg69, %142) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %143 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%32, %143) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    %144 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @AliasOp(%7, %144) {offset = 0 : index} : memref<25088xi8>, memref<4x256x14x14xi1>
    byre.compute @PTXOp(%142, %143, %144) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown49"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    %145 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%31, %145) {offset = 0 : index} : memref<401408xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%143, %84, %145) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %146 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %146) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%145, %arg73, %arg74, %146) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %147 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%48, %147) {offset = 0 : index} : memref<1179648xi8>, memref<4x256x14x14xf16>
    %148 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @AliasOp(%38, %148) {offset = 401408 : index} : memref<589824xi8>, memref<4x256x14x14xi1>
    byre.compute @PTXOp(%146, %139, %147, %148) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown51"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    %149 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%42, %149) {offset = 589824 : index} : memref<802816xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%147, %85, %149) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>
    %150 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %150) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%149, %arg88, %arg89, %150) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %151 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%26, %151) {offset = 0 : index} : memref<200704xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%147, %86, %151) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>
    %152 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %152) {offset = 1003520 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%151, %arg78, %arg79, %152) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %153 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%53, %153) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    %154 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @AliasOp(%52, %154) {offset = 1581056 : index} : memref<1605632xi8>, memref<4x512x7x7xi1>
    byre.compute @PTXOp(%152, %153, %154) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown54"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    %155 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%53, %155) {offset = 1380352 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%153, %87, %155) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %156 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %156) {offset = 1003520 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%155, %arg83, %arg84, %156) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %157 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%48, %157) {offset = 401408 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    %158 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @AliasOp(%53, %158) {offset = 1581056 : index} : memref<1605632xi8>, memref<4x512x7x7xi1>
    byre.compute @PTXOp(%156, %150, %157, %158) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown56"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    %159 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%48, %159) {offset = 602112 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%157, %88, %159) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %160 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %160) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%159, %arg93, %arg94, %160) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %161 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%48, %161) {offset = 802816 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    %162 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @AliasOp(%38, %162) {offset = 426496 : index} : memref<589824xi8>, memref<4x512x7x7xi1>
    byre.compute @PTXOp(%160, %161, %162) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown58"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    %163 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%24, %163) {offset = 0 : index} : memref<200704xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%161, %89, %163) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %164 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%25, %164) {offset = 0 : index} : memref<200704xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%163, %arg98, %arg99, %164) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %165 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %165) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    %166 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @AliasOp(%48, %166) {offset = 1003520 : index} : memref<1179648xi8>, memref<4x512x7x7xi1>
    byre.compute @PTXOp(%164, %157, %165, %166) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown60"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    %167 = memref.alloc() : memref<4x512xf16>
    byre.compute @AliasOp(%53, %167) {offset = 1593600 : index} : memref<1605632xi8>, memref<4x512xf16>
    byre.compute @ReduceSumOpf16f16(%165, %167) {dimensions = dense<[3, 2]> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512xf16>
    %168 = memref.alloc() : memref<4x512xf16>
    byre.compute @AliasOp(%1, %168) {offset = 0 : index} : memref<4096xi8>, memref<4x512xf16>
    byre.compute @PTXOp(%167, %168) {BlockSize.x = 128 : i32, GridSize.x = 16 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown61"} : memref<4x512xf16>, memref<4x512xf16>
    %169 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%54, %169) {offset = 802816 : index} : memref<1605632xi8>, memref<4x1000xf16>
    byre.compute @MatmulOpf16f16f16(%168, %91, %169) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>
    %170 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%52, %170) {offset = 1593600 : index} : memref<1605632xi8>, memref<4x1000xf16>
    byre.compute @PTXOp(%92, %169, %170) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown62"} : memref<1000xf16>, memref<4x1000xf16>, memref<4x1000xf16>
    %171 = memref.alloc() : memref<4xf16>
    byre.compute @AliasOp(%52, %171) {offset = 1601600 : index} : memref<1605632xi8>, memref<4xf16>
    byre.compute @ReduceMaxOpf16f16(%170, %171) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %172 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%53, %172) {offset = 1593600 : index} : memref<1605632xi8>, memref<4x1000xf16>
    %173 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%54, %173) {offset = 802816 : index} : memref<1605632xi8>, memref<4x1000xf16>
    byre.compute @PTXOp(%171, %170, %172, %173) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown63"} : memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf16>, memref<4x1000xf16>
    %174 = memref.alloc() : memref<4xf16>
    byre.compute @AliasOp(%53, %174) {offset = 1601600 : index} : memref<1605632xi8>, memref<4xf16>
    byre.compute @ReduceSumOpf16f16(%173, %174) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %175 = memref.alloc() : memref<4xf16>
    byre.compute @AliasOp(%54, %175) {offset = 802816 : index} : memref<1605632xi8>, memref<4xf16>
    byre.compute @PTXOp(%174, %175) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown64"} : memref<4xf16>, memref<4xf16>
    %176 = memref.alloc() : memref<4x1000xf16>
    byre.compute @AliasOp(%3, %176) {offset = 0 : index} : memref<8000xi8>, memref<4x1000xf16>
    %177 = memref.alloc() : memref<4x1000xf32>
    byre.compute @AliasOp(%4, %177) {offset = 0 : index} : memref<16000xi8>, memref<4x1000xf32>
    %178 = memref.alloc() : memref<4x1000xf32>
    byre.compute @AliasOp(%5, %178) {offset = 0 : index} : memref<16000xi8>, memref<4x1000xf32>
    byre.compute @PTXOp(%175, %172, %93, %90, %arg1, %176, %177, %178) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown65"} : memref<4xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>
    %179 = memref.alloc() : memref<4x512xf16>
    byre.compute @AliasOp(%54, %179) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512xf16>
    byre.compute @MatmulOpf16f16f16(%176, %91, %179) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>
    %180 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%25, %180) {offset = 0 : index} : memref<200704xi8>, memref<4x512x7x7xf16>
    byre.compute @PTXOp(%179, %166, %180) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [2 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown66"} : memref<4x512xf16>, memref<4x512x7x7xi1>, memref<4x512x7x7xf16>
    %181 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %181) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%163, %arg98, %180, %181, %arg163, %arg164) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %182 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%49, %182) {offset = 0 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%181, %89, %182) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %183 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%63, %183) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%161, %181, %183) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %184 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %184) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @PTXOp(%162, %182, %184) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown70"} : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>
    %185 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%49, %185) {offset = 0 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%159, %arg93, %184, %185, %arg160, %arg161) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %186 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %186) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%185, %88, %186) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %187 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%61, %187) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%157, %185, %187) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %188 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%48, %188) {offset = 401408 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @PTXOp(%180, %186, %158, %188) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown74"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>, memref<4x512x7x7xf16>
    %189 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%49, %189) {offset = 0 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%155, %arg83, %188, %189, %arg154, %arg155) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %190 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %190) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%189, %87, %190) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %191 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @AliasOp(%62, %191) {offset = 0 : index} : memref<4718592xi8>, memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%153, %189, %191) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %192 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%53, %192) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @PTXOp(%154, %190, %192) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown78"} : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>
    %193 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%54, %193) {offset = 802816 : index} : memref<1605632xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%151, %arg78, %192, %193, %arg151, %arg152) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %194 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%53, %194) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%193, %86, %194) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x3x3xf16>, memref<4x256x14x14xf16>
    %195 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @AliasOp(%60, %195) {offset = 0 : index} : memref<2359296xi8>, memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%147, %193, %195) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x3x3xf16>
    %196 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @AliasOp(%49, %196) {offset = 0 : index} : memref<1179648xi8>, memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%149, %arg88, %188, %196, %arg157, %arg158) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %197 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %197) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%196, %85, %197) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x1x1xf16>, memref<4x256x14x14xf16>
    %198 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @AliasOp(%27, %198) {offset = 0 : index} : memref<262144xi8>, memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%147, %196, %198) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x1x1xf16>
    %199 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%48, %199) {offset = 0 : index} : memref<1179648xi8>, memref<4x256x14x14xf16>
    byre.compute @PTXOp(%197, %194, %148, %199) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown85"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>, memref<4x256x14x14xf16>
    %200 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%53, %200) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%145, %arg73, %199, %200, %arg148, %arg149) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %201 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %201) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%200, %84, %201) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %202 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%50, %202) {offset = 0 : index} : memref<1179648xi8>, memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%143, %200, %202) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %203 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%53, %203) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @PTXOp(%144, %201, %203) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown89"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %204 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %204) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%141, %arg68, %203, %204, %arg145, %arg146) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %205 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%53, %205) {offset = 1179648 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%204, %83, %205) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %206 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%49, %206) {offset = 0 : index} : memref<1179648xi8>, memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%139, %204, %206) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %207 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%52, %207) {offset = 0 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @PTXOp(%199, %205, %140, %207) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown93"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>, memref<4x256x14x14xf16>
    %208 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %208) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%137, %arg58, %207, %208, %arg139, %arg140) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %209 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%52, %209) {offset = 401408 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%208, %82, %209) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %210 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @AliasOp(%48, %210) {offset = 0 : index} : memref<1179648xi8>, memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%135, %208, %210) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %211 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%53, %211) {offset = 0 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @PTXOp(%136, %209, %211) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown97"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %212 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %212) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%133, %arg53, %211, %212, %arg136, %arg137) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %213 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %213) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%212, %81, %213) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x3x3xf16>, memref<4x128x28x28xf16>
    %214 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @AliasOp(%38, %214) {offset = 0 : index} : memref<589824xi8>, memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%129, %212, %214) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x3x3xf16>
    %215 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @AliasOp(%54, %215) {offset = 802816 : index} : memref<1605632xi8>, memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%131, %arg63, %207, %215, %arg142, %arg143) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %216 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %216) {offset = 802816 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%215, %80, %216) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x1x1xf16>, memref<4x128x28x28xf16>
    %217 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @AliasOp(%36, %217) {offset = 0 : index} : memref<401408xi8>, memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%129, %215, %217) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x1x1xf16>
    %218 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%42, %218) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @PTXOp(%216, %213, %130, %218) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown104"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>, memref<4x128x28x28xf16>
    %219 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %219) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%127, %arg48, %218, %219, %arg133, %arg134) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %220 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %220) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%219, %79, %220) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %221 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%40, %221) {offset = 0 : index} : memref<802816xi8>, memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%125, %219, %221) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %222 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %222) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @PTXOp(%126, %220, %222) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown108"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %223 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %223) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%123, %arg43, %222, %223, %arg130, %arg131) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %224 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %224) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%223, %78, %224) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %225 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%45, %225) {offset = 0 : index} : memref<802816xi8>, memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%121, %223, %225) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %226 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%52, %226) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @PTXOp(%218, %224, %122, %226) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown112"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>, memref<4x128x28x28xf16>
    %227 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %227) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%119, %arg33, %226, %227, %arg124, %arg125) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %228 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %228) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%227, %77, %228) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %229 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @AliasOp(%46, %229) {offset = 0 : index} : memref<802816xi8>, memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%117, %227, %229) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %230 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%54, %230) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @PTXOp(%118, %228, %230) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown116"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %231 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%53, %231) {offset = 0 : index} : memref<1605632xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%115, %arg28, %230, %231, %arg121, %arg122) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %232 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%54, %232) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%231, %76, %232) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x3x3xf16>, memref<4x64x56x56xf16>
    %233 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @AliasOp(%46, %233) {offset = 294912 : index} : memref<802816xi8>, memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%111, %231, %233) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x3x3xf16>
    %234 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @AliasOp(%47, %234) {offset = 0 : index} : memref<802816xi8>, memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%113, %arg38, %226, %234, %arg127, %arg128) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %235 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%53, %235) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%234, %75, %235) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x1x1xf16>, memref<4x64x56x56xf16>
    %236 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @AliasOp(%45, %236) {offset = 294912 : index} : memref<802816xi8>, memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%111, %234, %236) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x1x1xf16>
    %237 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%52, %237) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @PTXOp(%235, %232, %112, %237) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown123"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>, memref<4x64x56x56xf16>
    %238 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%53, %238) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%109, %arg23, %237, %238, %arg118, %arg119) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %239 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%54, %239) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%238, %74, %239) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %240 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%47, %240) {offset = 0 : index} : memref<802816xi8>, memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%107, %238, %240) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %241 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %241) {offset = 3211264 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @PTXOp(%108, %239, %241) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown127"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %242 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%55, %242) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%105, %arg18, %241, %242, %arg115, %arg116) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %243 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %243) {offset = 3211264 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%242, %73, %243) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %244 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%54, %244) {offset = 0 : index} : memref<1605632xi8>, memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%103, %242, %244) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %245 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%56, %245) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @PTXOp(%237, %243, %104, %245) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown131"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>, memref<4x64x56x56xf16>
    %246 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%57, %246) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%101, %arg13, %245, %246, %arg112, %arg113) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %247 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %247) {offset = 1605632 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%246, %72, %247) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %248 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%55, %248) {offset = 0 : index} : memref<1605632xi8>, memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%99, %246, %248) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %249 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%58, %249) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @PTXOp(%100, %247, %249) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown135"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %250 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%64, %250) {offset = 1605632 : index} : memref<6422528xi8>, memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%97, %arg8, %249, %250, %arg109, %arg110) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %251 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%58, %251) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%250, %71, %251) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %252 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @AliasOp(%57, %252) {offset = 0 : index} : memref<1605632xi8>, memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%96, %250, %252) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %253 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @AliasOp(%59, %253) {offset = 0 : index} : memref<1605632xi8>, memref<4x64x56x56xf16>
    byre.compute @PTXOp(%245, %251, %253) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown139"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %254 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%64, %254) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    byre.compute @PoolMaxGradOpf16f16f16(%94, %253, %254) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<4x64x112x112xf16>
    %255 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%66, %255) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    byre.compute @PTXOp(%95, %254, %255) {BlockSize.x = 128 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown140"} : memref<4x64x112x112xi1>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>
    %256 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @AliasOp(%64, %256) {offset = 0 : index} : memref<6422528xi8>, memref<4x64x112x112xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%69, %arg3, %255, %256, %arg106, %arg107) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %257 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @AliasOp(%66, %257) {offset = 0 : index} : memref<6422528xi8>, memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%67, %256, %257) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<64x3x7x7xf16>
    %258 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%66, %258) {offset = 18816 : index} : memref<6422528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%177, %258) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<4x1000xf32>, memref<f32>
    byre.compute @PTXOp(%258, %arg104) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown143"} : memref<f32>, memref<f32>
    byre.compute @PTXOp(%257, %arg105) {BlockSize.x = 128 : i32, GridSize.x = 74 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown144"} : memref<64x3x7x7xf16>, memref<64x3x7x7xf32>
    byre.compute @PTXOp(%252, %arg108) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown145"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%248, %arg111) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown146"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%244, %arg114) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown147"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%240, %arg117) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown148"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%233, %arg120) {BlockSize.x = 128 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown149"} : memref<128x64x3x3xf16>, memref<128x64x3x3xf32>
    byre.compute @PTXOp(%229, %arg123) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%236, %arg126) {BlockSize.x = 128 : i32, GridSize.x = 64 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown151"} : memref<128x64x1x1xf16>, memref<128x64x1x1xf32>
    byre.compute @PTXOp(%225, %arg129) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown152"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%221, %arg132) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown153"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%214, %arg135) {BlockSize.x = 128 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown154"} : memref<256x128x3x3xf16>, memref<256x128x3x3xf32>
    byre.compute @PTXOp(%210, %arg138) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown155"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%217, %arg141) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown156"} : memref<256x128x1x1xf16>, memref<256x128x1x1xf32>
    byre.compute @PTXOp(%206, %arg144) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown157"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%202, %arg147) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown158"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%195, %arg150) {BlockSize.x = 128 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown159"} : memref<512x256x3x3xf16>, memref<512x256x3x3xf32>
    byre.compute @PTXOp(%191, %arg153) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown160"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%198, %arg156) {BlockSize.x = 128 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown161"} : memref<512x256x1x1xf16>, memref<512x256x1x1xf32>
    byre.compute @PTXOp(%187, %arg159) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown162"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%183, %arg162) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown163"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    %259 = memref.alloc() : memref<1000x512xf16>
    byre.compute @AliasOp(%66, %259) {offset = 0 : index} : memref<6422528xi8>, memref<1000x512xf16>
    byre.compute @MatmulOpf16f16f16(%168, %176, %259) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16>, memref<4x1000xf16>, memref<1000x512xf16>
    byre.compute @PTXOp(%259, %arg165) {BlockSize.x = 128 : i32, GridSize.x = 4000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown165"} : memref<1000x512xf16>, memref<1000x512xf32>
    %260 = memref.alloc() : memref<1000xf32>
    byre.compute @AliasOp(%66, %260) {offset = 0 : index} : memref<6422528xi8>, memref<1000xf32>
    byre.compute @ReduceSumOpf32f32(%178, %260) {dimensions = dense<0> : tensor<1xi64>} : memref<4x1000xf32>, memref<1000xf32>
    byre.compute @PTXOp(%260, %arg166) {BlockSize.x = 128 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown166"} : memref<1000xf32>, memref<1000xf32>
    return
  }
}

