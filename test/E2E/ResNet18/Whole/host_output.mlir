// RUN: byteir-opt %s | FileCheck %s

// CHECK-LABEL: func @main
module @IrToMhlo.2452 attributes {byre.container_module, gpu.container_module} {
  func @main(%arg0: memref<4x3x224x224xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<4x1000xf32> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<64x3x7x7xf32> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<64xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<64xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<64xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<64xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<64x64x3x3xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<64xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<64xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<64xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<64xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<64x64x3x3xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<64xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<64xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<64xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<64xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<64x64x3x3xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<64xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<64xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<64xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<64xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<64x64x3x3xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<64xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<64xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<64xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<64xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x64x3x3xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128x128x3x3xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<128xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x64x1x1xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128x128x3x3xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<128xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<128xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<128x128x3x3xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<128xf32> {byre.argname = "Input48", byre.argtype = 1 : i32}, %arg49: memref<128xf32> {byre.argname = "Input49", byre.argtype = 1 : i32}, %arg50: memref<128xf32> {byre.argname = "Input50", byre.argtype = 1 : i32}, %arg51: memref<128xf32> {byre.argname = "Input51", byre.argtype = 1 : i32}, %arg52: memref<256x128x3x3xf32> {byre.argname = "Input52", byre.argtype = 1 : i32}, %arg53: memref<256xf32> {byre.argname = "Input53", byre.argtype = 1 : i32}, %arg54: memref<256xf32> {byre.argname = "Input54", byre.argtype = 1 : i32}, %arg55: memref<256xf32> {byre.argname = "Input55", byre.argtype = 1 : i32}, %arg56: memref<256xf32> {byre.argname = "Input56", byre.argtype = 1 : i32}, %arg57: memref<256x256x3x3xf32> {byre.argname = "Input57", byre.argtype = 1 : i32}, %arg58: memref<256xf32> {byre.argname = "Input58", byre.argtype = 1 : i32}, %arg59: memref<256xf32> {byre.argname = "Input59", byre.argtype = 1 : i32}, %arg60: memref<256xf32> {byre.argname = "Input60", byre.argtype = 1 : i32}, %arg61: memref<256xf32> {byre.argname = "Input61", byre.argtype = 1 : i32}, %arg62: memref<256x128x1x1xf32> {byre.argname = "Input62", byre.argtype = 1 : i32}, %arg63: memref<256xf32> {byre.argname = "Input63", byre.argtype = 1 : i32}, %arg64: memref<256xf32> {byre.argname = "Input64", byre.argtype = 1 : i32}, %arg65: memref<256xf32> {byre.argname = "Input65", byre.argtype = 1 : i32}, %arg66: memref<256xf32> {byre.argname = "Input66", byre.argtype = 1 : i32}, %arg67: memref<256x256x3x3xf32> {byre.argname = "Input67", byre.argtype = 1 : i32}, %arg68: memref<256xf32> {byre.argname = "Input68", byre.argtype = 1 : i32}, %arg69: memref<256xf32> {byre.argname = "Input69", byre.argtype = 1 : i32}, %arg70: memref<256xf32> {byre.argname = "Input70", byre.argtype = 1 : i32}, %arg71: memref<256xf32> {byre.argname = "Input71", byre.argtype = 1 : i32}, %arg72: memref<256x256x3x3xf32> {byre.argname = "Input72", byre.argtype = 1 : i32}, %arg73: memref<256xf32> {byre.argname = "Input73", byre.argtype = 1 : i32}, %arg74: memref<256xf32> {byre.argname = "Input74", byre.argtype = 1 : i32}, %arg75: memref<256xf32> {byre.argname = "Input75", byre.argtype = 1 : i32}, %arg76: memref<256xf32> {byre.argname = "Input76", byre.argtype = 1 : i32}, %arg77: memref<512x256x3x3xf32> {byre.argname = "Input77", byre.argtype = 1 : i32}, %arg78: memref<512xf32> {byre.argname = "Input78", byre.argtype = 1 : i32}, %arg79: memref<512xf32> {byre.argname = "Input79", byre.argtype = 1 : i32}, %arg80: memref<512xf32> {byre.argname = "Input80", byre.argtype = 1 : i32}, %arg81: memref<512xf32> {byre.argname = "Input81", byre.argtype = 1 : i32}, %arg82: memref<512x512x3x3xf32> {byre.argname = "Input82", byre.argtype = 1 : i32}, %arg83: memref<512xf32> {byre.argname = "Input83", byre.argtype = 1 : i32}, %arg84: memref<512xf32> {byre.argname = "Input84", byre.argtype = 1 : i32}, %arg85: memref<512xf32> {byre.argname = "Input85", byre.argtype = 1 : i32}, %arg86: memref<512xf32> {byre.argname = "Input86", byre.argtype = 1 : i32}, %arg87: memref<512x256x1x1xf32> {byre.argname = "Input87", byre.argtype = 1 : i32}, %arg88: memref<512xf32> {byre.argname = "Input88", byre.argtype = 1 : i32}, %arg89: memref<512xf32> {byre.argname = "Input89", byre.argtype = 1 : i32}, %arg90: memref<512xf32> {byre.argname = "Input90", byre.argtype = 1 : i32}, %arg91: memref<512xf32> {byre.argname = "Input91", byre.argtype = 1 : i32}, %arg92: memref<512x512x3x3xf32> {byre.argname = "Input92", byre.argtype = 1 : i32}, %arg93: memref<512xf32> {byre.argname = "Input93", byre.argtype = 1 : i32}, %arg94: memref<512xf32> {byre.argname = "Input94", byre.argtype = 1 : i32}, %arg95: memref<512xf32> {byre.argname = "Input95", byre.argtype = 1 : i32}, %arg96: memref<512xf32> {byre.argname = "Input96", byre.argtype = 1 : i32}, %arg97: memref<512x512x3x3xf32> {byre.argname = "Input97", byre.argtype = 1 : i32}, %arg98: memref<512xf32> {byre.argname = "Input98", byre.argtype = 1 : i32}, %arg99: memref<512xf32> {byre.argname = "Input99", byre.argtype = 1 : i32}, %arg100: memref<512xf32> {byre.argname = "Input100", byre.argtype = 1 : i32}, %arg101: memref<512xf32> {byre.argname = "Input101", byre.argtype = 1 : i32}, %arg102: memref<1000x512xf32> {byre.argname = "Input102", byre.argtype = 1 : i32}, %arg103: memref<1000xf32> {byre.argname = "Input103", byre.argtype = 1 : i32}, %arg104: memref<f32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg105: memref<64x3x7x7xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg106: memref<64xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg107: memref<64xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg108: memref<64x64x3x3xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg109: memref<64xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg110: memref<64xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg111: memref<64x64x3x3xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg112: memref<64xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg113: memref<64xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg114: memref<64x64x3x3xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg115: memref<64xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg116: memref<64xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg117: memref<64x64x3x3xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg118: memref<64xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg119: memref<64xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg120: memref<128x64x3x3xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg121: memref<128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg122: memref<128xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg123: memref<128x128x3x3xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg124: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg125: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg126: memref<128x64x1x1xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg127: memref<128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg128: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg129: memref<128x128x3x3xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg130: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg131: memref<128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg132: memref<128x128x3x3xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg133: memref<128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg134: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg135: memref<256x128x3x3xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg136: memref<256xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg137: memref<256xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg138: memref<256x256x3x3xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg139: memref<256xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg140: memref<256xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg141: memref<256x128x1x1xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg142: memref<256xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg143: memref<256xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg144: memref<256x256x3x3xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg145: memref<256xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg146: memref<256xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg147: memref<256x256x3x3xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}, %arg148: memref<256xf32> {byre.argname = "Output44", byre.argtype = 2 : i32}, %arg149: memref<256xf32> {byre.argname = "Output45", byre.argtype = 2 : i32}, %arg150: memref<512x256x3x3xf32> {byre.argname = "Output46", byre.argtype = 2 : i32}, %arg151: memref<512xf32> {byre.argname = "Output47", byre.argtype = 2 : i32}, %arg152: memref<512xf32> {byre.argname = "Output48", byre.argtype = 2 : i32}, %arg153: memref<512x512x3x3xf32> {byre.argname = "Output49", byre.argtype = 2 : i32}, %arg154: memref<512xf32> {byre.argname = "Output50", byre.argtype = 2 : i32}, %arg155: memref<512xf32> {byre.argname = "Output51", byre.argtype = 2 : i32}, %arg156: memref<512x256x1x1xf32> {byre.argname = "Output52", byre.argtype = 2 : i32}, %arg157: memref<512xf32> {byre.argname = "Output53", byre.argtype = 2 : i32}, %arg158: memref<512xf32> {byre.argname = "Output54", byre.argtype = 2 : i32}, %arg159: memref<512x512x3x3xf32> {byre.argname = "Output55", byre.argtype = 2 : i32}, %arg160: memref<512xf32> {byre.argname = "Output56", byre.argtype = 2 : i32}, %arg161: memref<512xf32> {byre.argname = "Output57", byre.argtype = 2 : i32}, %arg162: memref<512x512x3x3xf32> {byre.argname = "Output58", byre.argtype = 2 : i32}, %arg163: memref<512xf32> {byre.argname = "Output59", byre.argtype = 2 : i32}, %arg164: memref<512xf32> {byre.argname = "Output60", byre.argtype = 2 : i32}, %arg165: memref<1000x512xf32> {byre.argname = "Output61", byre.argtype = 2 : i32}, %arg166: memref<1000xf32> {byre.argname = "Output62", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "test/test_files/resnet18_fw_bw_device.ptx"} {
    %0 = memref.alloc() : memref<1000xf32>
    %1 = memref.alloc() : memref<f32>
    %2 = memref.alloc() : memref<4x64x112x112xf16>
    %3 = memref.alloc() : memref<4x512xf16>
    %4 = memref.alloc() : memref<4xf16>
    %5 = memref.alloc() : memref<4xf16>
    %6 = memref.alloc() : memref<4x1000xf16>
    %7 = memref.alloc() : memref<4x512xf16>
    %8 = memref.alloc() : memref<4x512x7x7xf16>
    %9 = memref.alloc() : memref<4x512x7x7xf16>
    %10 = memref.alloc() : memref<4x512x7x7xf16>
    %11 = memref.alloc() : memref<4x512x7x7xf16>
    %12 = memref.alloc() : memref<4x512x7x7xf16>
    %13 = memref.alloc() : memref<4x256x14x14xf16>
    %14 = memref.alloc() : memref<4x256x14x14xf16>
    %15 = memref.alloc() : memref<4x256x14x14xf16>
    %16 = memref.alloc() : memref<4x256x14x14xf16>
    %17 = memref.alloc() : memref<4x256x14x14xf16>
    %18 = memref.alloc() : memref<4x128x28x28xf16>
    %19 = memref.alloc() : memref<4x128x28x28xf16>
    %20 = memref.alloc() : memref<4x128x28x28xf16>
    %21 = memref.alloc() : memref<4x128x28x28xf16>
    %22 = memref.alloc() : memref<4x128x28x28xf16>
    %23 = memref.alloc() : memref<4x64x56x56xf16>
    %24 = memref.alloc() : memref<4x64x56x56xf16>
    %25 = memref.alloc() : memref<4x64x56x56xf16>
    %26 = memref.alloc() : memref<4x64x56x56xf16>
    %27 = memref.alloc() : memref<4x64x56x56xf16>
    %28 = memref.alloc() : memref<4xf16>
    %29 = memref.alloc() : memref<4x64x112x112xf16>
    %30 = memref.alloc() : memref<4x3x224x224xf16>
    byre.compute @PTXOp(%arg0, %30) {BlockSize.x = 128 : i32, GridSize.x = 4704 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown0_kernel"} : memref<4x3x224x224xf32>, memref<4x3x224x224xf16>
    %31 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @PTXOp(%arg2, %31) {BlockSize.x = 128 : i32, GridSize.x = 74 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown1_kernel"} : memref<64x3x7x7xf32>, memref<64x3x7x7xf16>
    byre.compute @ConvOpf16f16f16(%30, %31, %29) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<3> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<64x3x7x7xf16>, memref<4x64x112x112xf16>
    %32 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%29, %arg3, %arg4, %32) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x112x112xf16>
    %33 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg7, %33) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown3_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %34 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg12, %34) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown4_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %35 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg17, %35) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown5_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %36 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @PTXOp(%arg22, %36) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown6_kernel"} : memref<64x64x3x3xf32>, memref<64x64x3x3xf16>
    %37 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @PTXOp(%arg37, %37) {BlockSize.x = 128 : i32, GridSize.x = 64 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown7_kernel"} : memref<128x64x1x1xf32>, memref<128x64x1x1xf16>
    %38 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @PTXOp(%arg27, %38) {BlockSize.x = 128 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown8_kernel"} : memref<128x64x3x3xf32>, memref<128x64x3x3xf16>
    %39 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg32, %39) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown9_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %40 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg42, %40) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown10_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %41 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @PTXOp(%arg47, %41) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown11_kernel"} : memref<128x128x3x3xf32>, memref<128x128x3x3xf16>
    %42 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @PTXOp(%arg62, %42) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown12_kernel"} : memref<256x128x1x1xf32>, memref<256x128x1x1xf16>
    %43 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @PTXOp(%arg52, %43) {BlockSize.x = 128 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown13_kernel"} : memref<256x128x3x3xf32>, memref<256x128x3x3xf16>
    %44 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg57, %44) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown14_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %45 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg67, %45) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown15_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %46 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @PTXOp(%arg72, %46) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown16_kernel"} : memref<256x256x3x3xf32>, memref<256x256x3x3xf16>
    %47 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @PTXOp(%arg87, %47) {BlockSize.x = 128 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown17_kernel"} : memref<512x256x1x1xf32>, memref<512x256x1x1xf16>
    %48 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @PTXOp(%arg77, %48) {BlockSize.x = 128 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown18_kernel"} : memref<512x256x3x3xf32>, memref<512x256x3x3xf16>
    %49 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg82, %49) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown19_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %50 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg92, %50) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown20_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %51 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @PTXOp(%arg97, %51) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown21_kernel"} : memref<512x512x3x3xf32>, memref<512x512x3x3xf16>
    %52 = memref.alloc() : memref<4x1000xf16>
    byre.compute @PTXOp(%arg1, %52) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown22_kernel"} : memref<4x1000xf32>, memref<4x1000xf16>
    %53 = memref.alloc() : memref<1000x512xf16>
    byre.compute @PTXOp(%arg102, %53) {BlockSize.x = 128 : i32, GridSize.x = 4000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown23_kernel"} : memref<1000x512xf32>, memref<1000x512xf16>
    %54 = memref.alloc() : memref<1000xf16>
    byre.compute @PTXOp(%arg103, %54) {BlockSize.x = 128 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown24_kernel"} : memref<1000xf32>, memref<1000xf16>
    byre.compute @ReduceSumOpf16f16(%52, %28) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %55 = memref.alloc() : memref<4x64x112x112xf16>
    %56 = memref.alloc() : memref<4x64x112x112xi1>
    byre.compute @PTXOp(%32, %55, %56) {BlockSize.x = 128 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown25_kernel"} : memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<4x64x112x112xi1>
    byre.compute @PoolMaxOpf16f16(%55, %27) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>
    byre.compute @ConvOpf16f16f16(%27, %33, %26) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %57 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%26, %arg8, %arg9, %57) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %58 = memref.alloc() : memref<4x64x56x56xf16>
    %59 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @PTXOp(%57, %58, %59) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown27_kernel"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    byre.compute @ConvOpf16f16f16(%58, %34, %25) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %60 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%25, %arg13, %arg14, %60) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %61 = memref.alloc() : memref<4x64x56x56xf16>
    %62 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @PTXOp(%60, %27, %61, %62) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown29_kernel"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    byre.compute @ConvOpf16f16f16(%61, %35, %24) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %63 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%24, %arg18, %arg19, %63) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %64 = memref.alloc() : memref<4x64x56x56xf16>
    %65 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @PTXOp(%63, %64, %65) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown31_kernel"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    byre.compute @ConvOpf16f16f16(%64, %36, %23) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %66 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%23, %arg23, %arg24, %66) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>, memref<4x64x56x56xf16>
    %67 = memref.alloc() : memref<4x64x56x56xf16>
    %68 = memref.alloc() : memref<4x64x56x56xi1>
    byre.compute @PTXOp(%66, %61, %67, %68) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown33_kernel"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xi1>
    byre.compute @ConvOpf16f16f16(%67, %37, %22) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x1x1xf16>, memref<4x128x28x28xf16>
    %69 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%22, %arg38, %arg39, %69) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    byre.compute @ConvOpf16f16f16(%67, %38, %21) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<128x64x3x3xf16>, memref<4x128x28x28xf16>
    %70 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%21, %arg28, %arg29, %70) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %71 = memref.alloc() : memref<4x128x28x28xf16>
    %72 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @PTXOp(%70, %71, %72) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown36_kernel"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    byre.compute @ConvOpf16f16f16(%71, %39, %20) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %73 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%20, %arg33, %arg34, %73) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %74 = memref.alloc() : memref<4x128x28x28xf16>
    %75 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @PTXOp(%73, %69, %74, %75) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown38_kernel"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    byre.compute @ConvOpf16f16f16(%74, %40, %19) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %76 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%19, %arg43, %arg44, %76) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %77 = memref.alloc() : memref<4x128x28x28xf16>
    %78 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @PTXOp(%76, %77, %78) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown40_kernel"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    byre.compute @ConvOpf16f16f16(%77, %41, %18) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %79 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%18, %arg48, %arg49, %79) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>, memref<4x128x28x28xf16>
    %80 = memref.alloc() : memref<4x128x28x28xf16>
    %81 = memref.alloc() : memref<4x128x28x28xi1>
    byre.compute @PTXOp(%79, %74, %80, %81) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown42_kernel"} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xi1>
    byre.compute @ConvOpf16f16f16(%80, %42, %17) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x1x1xf16>, memref<4x256x14x14xf16>
    %82 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%17, %arg63, %arg64, %82) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    byre.compute @ConvOpf16f16f16(%80, %43, %16) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<256x128x3x3xf16>, memref<4x256x14x14xf16>
    %83 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%16, %arg53, %arg54, %83) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %84 = memref.alloc() : memref<4x256x14x14xf16>
    %85 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @PTXOp(%83, %84, %85) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown45_kernel"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    byre.compute @ConvOpf16f16f16(%84, %44, %15) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %86 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%15, %arg58, %arg59, %86) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %87 = memref.alloc() : memref<4x256x14x14xf16>
    %88 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @PTXOp(%86, %82, %87, %88) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown47_kernel"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    byre.compute @ConvOpf16f16f16(%87, %45, %14) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %89 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%14, %arg68, %arg69, %89) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %90 = memref.alloc() : memref<4x256x14x14xf16>
    %91 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @PTXOp(%89, %90, %91) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown49_kernel"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    byre.compute @ConvOpf16f16f16(%90, %46, %13) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %92 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%13, %arg73, %arg74, %92) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>, memref<4x256x14x14xf16>
    %93 = memref.alloc() : memref<4x256x14x14xf16>
    %94 = memref.alloc() : memref<4x256x14x14xi1>
    byre.compute @PTXOp(%92, %87, %93, %94) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown51_kernel"} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xi1>
    byre.compute @ConvOpf16f16f16(%93, %47, %12) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x1x1xf16>, memref<4x512x7x7xf16>
    %95 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%12, %arg88, %arg89, %95) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    byre.compute @ConvOpf16f16f16(%93, %48, %11) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<512x256x3x3xf16>, memref<4x512x7x7xf16>
    %96 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%11, %arg78, %arg79, %96) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %97 = memref.alloc() : memref<4x512x7x7xf16>
    %98 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @PTXOp(%96, %97, %98) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown54_kernel"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    byre.compute @ConvOpf16f16f16(%97, %49, %10) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %99 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%10, %arg83, %arg84, %99) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %100 = memref.alloc() : memref<4x512x7x7xf16>
    %101 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @PTXOp(%99, %95, %100, %101) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown56_kernel"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    byre.compute @ConvOpf16f16f16(%100, %50, %9) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %102 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%9, %arg93, %arg94, %102) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %103 = memref.alloc() : memref<4x512x7x7xf16>
    %104 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @PTXOp(%102, %103, %104) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown58_kernel"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    byre.compute @ConvOpf16f16f16(%103, %51, %8) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", lhs_dilation = dense<1> : tensor<2xi64>, output_layout = "NCHW", padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %105 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormTrainingOpf16f32f32f16(%8, %arg98, %arg99, %105) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>, memref<4x512x7x7xf16>
    %106 = memref.alloc() : memref<4x512x7x7xf16>
    %107 = memref.alloc() : memref<4x512x7x7xi1>
    byre.compute @PTXOp(%105, %100, %106, %107) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown60_kernel"} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xi1>
    byre.compute @ReduceSumOpf16f16(%106, %7) {dimensions = dense<[3, 2]> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512xf16>
    %108 = memref.alloc() : memref<4x512xf16>
    byre.compute @PTXOp(%7, %108) {BlockSize.x = 128 : i32, GridSize.x = 16 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown61_kernel"} : memref<4x512xf16>, memref<4x512xf16>
    byre.compute @MatmulOpf16f16f16(%108, %53, %6) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<4x512xf16>, memref<1000x512xf16>, memref<4x1000xf16>
    %109 = memref.alloc() : memref<4x1000xf16>
    byre.compute @PTXOp(%6, %54, %109) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32], kernel_name = "Unknown62_kernel"} : memref<4x1000xf16>, memref<1000xf16>, memref<4x1000xf16>
    byre.compute @ReduceMaxOpf16f16(%109, %5) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %110 = memref.alloc() : memref<4x1000xf16>
    %111 = memref.alloc() : memref<4x1000xf16>
    byre.compute @PTXOp(%109, %5, %110, %111) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown63_kernel"} : memref<4x1000xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf16>
    byre.compute @ReduceSumOpf16f16(%111, %4) {dimensions = dense<1> : tensor<1xi64>} : memref<4x1000xf16>, memref<4xf16>
    %112 = memref.alloc() : memref<4xf16>
    byre.compute @PTXOp(%4, %112) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown64_kernel"} : memref<4xf16>, memref<4xf16>
    %113 = memref.alloc() : memref<4x1000xf16>
    %114 = memref.alloc() : memref<4x1000xf32>
    %115 = memref.alloc() : memref<4x1000xf32>
    byre.compute @PTXOp(%52, %110, %112, %28, %113, %arg1, %114, %115) {BlockSize.x = 128 : i32, GridSize.x = 32 : i32, arg_ranks = [2 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown65_kernel"} : memref<4x1000xf16>, memref<4x1000xf16>, memref<4xf16>, memref<4xf16>, memref<4x1000xf16>, memref<4x1000xf32>, memref<4x1000xf32>, memref<4x1000xf32>
    byre.compute @MatmulOpf16f16f16(%113, %53, %3) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<4x1000xf16>, memref<1000x512xf16>, memref<4x512xf16>
    %116 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @PTXOp(%107, %3, %116) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 2 : i32, 4 : i32], kernel_name = "Unknown66_kernel"} : memref<4x512x7x7xi1>, memref<4x512xf16>, memref<4x512x7x7xf16>
    %117 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%8, %arg98, %116, %117, %arg163, %arg164) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %118 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%117, %51, %118) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %119 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%103, %117, %119) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %120 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @PTXOp(%104, %118, %120) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown70_kernel"} : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>
    %121 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%9, %arg93, %120, %121, %arg160, %arg161) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %122 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%121, %50, %122) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %123 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%100, %121, %123) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %124 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @PTXOp(%101, %116, %122, %124) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown74_kernel"} : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>
    %125 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%10, %arg83, %124, %125, %arg154, %arg155) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %126 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%125, %49, %126) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x512x3x3xf16>, memref<4x512x7x7xf16>
    %127 = memref.alloc() : memref<512x512x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%97, %125, %127) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512x512x3x3xf16>
    %128 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @PTXOp(%98, %126, %128) {BlockSize.x = 128 : i32, GridSize.x = 784 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown78_kernel"} : memref<4x512x7x7xi1>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>
    %129 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%11, %arg78, %128, %129, %arg151, %arg152) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %130 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%129, %48, %130) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x3x3xf16>, memref<4x256x14x14xf16>
    %131 = memref.alloc() : memref<512x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%93, %129, %131) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x3x3xf16>
    %132 = memref.alloc() : memref<4x512x7x7xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%12, %arg88, %124, %132, %arg157, %arg158) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x512x7x7xf16>, memref<512xf32>, memref<4x512x7x7xf16>, memref<4x512x7x7xf16>, memref<512xf32>, memref<512xf32>
    %133 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%132, %47, %133) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x512x7x7xf16>, memref<512x256x1x1xf16>, memref<4x256x14x14xf16>
    %134 = memref.alloc() : memref<512x256x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%93, %132, %134) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x512x7x7xf16>, memref<512x256x1x1xf16>
    %135 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @PTXOp(%94, %133, %130, %135) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown85_kernel"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %136 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%13, %arg73, %135, %136, %arg148, %arg149) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %137 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%136, %46, %137) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %138 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%90, %136, %138) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %139 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @PTXOp(%91, %137, %139) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown89_kernel"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %140 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%14, %arg68, %139, %140, %arg145, %arg146) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %141 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%140, %45, %141) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %142 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%87, %140, %142) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %143 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @PTXOp(%88, %135, %141, %143) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown93_kernel"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %144 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%15, %arg58, %143, %144, %arg139, %arg140) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %145 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%144, %44, %145) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x256x3x3xf16>, memref<4x256x14x14xf16>
    %146 = memref.alloc() : memref<256x256x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%84, %144, %146) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256x256x3x3xf16>
    %147 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @PTXOp(%85, %145, %147) {BlockSize.x = 128 : i32, GridSize.x = 1568 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown97_kernel"} : memref<4x256x14x14xi1>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>
    %148 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%16, %arg53, %147, %148, %arg136, %arg137) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %149 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%148, %43, %149) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x3x3xf16>, memref<4x128x28x28xf16>
    %150 = memref.alloc() : memref<256x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%80, %148, %150) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x3x3xf16>
    %151 = memref.alloc() : memref<4x256x14x14xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%17, %arg63, %143, %151, %arg142, %arg143) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x256x14x14xf16>, memref<256xf32>, memref<4x256x14x14xf16>, memref<4x256x14x14xf16>, memref<256xf32>, memref<256xf32>
    %152 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%151, %42, %152) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x256x14x14xf16>, memref<256x128x1x1xf16>, memref<4x128x28x28xf16>
    %153 = memref.alloc() : memref<256x128x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%80, %151, %153) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x256x14x14xf16>, memref<256x128x1x1xf16>
    %154 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @PTXOp(%81, %152, %149, %154) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown104_kernel"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %155 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%18, %arg48, %154, %155, %arg133, %arg134) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %156 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%155, %41, %156) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %157 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%77, %155, %157) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %158 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @PTXOp(%78, %156, %158) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown108_kernel"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %159 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%19, %arg43, %158, %159, %arg130, %arg131) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %160 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%159, %40, %160) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %161 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%74, %159, %161) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %162 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @PTXOp(%75, %154, %160, %162) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown112_kernel"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %163 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%20, %arg33, %162, %163, %arg124, %arg125) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %164 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%163, %39, %164) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x128x3x3xf16>, memref<4x128x28x28xf16>
    %165 = memref.alloc() : memref<128x128x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%71, %163, %165) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128x128x3x3xf16>
    %166 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @PTXOp(%72, %164, %166) {BlockSize.x = 128 : i32, GridSize.x = 3136 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown116_kernel"} : memref<4x128x28x28xi1>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>
    %167 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%21, %arg28, %166, %167, %arg121, %arg122) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %168 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%167, %38, %168) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x3x3xf16>, memref<4x64x56x56xf16>
    %169 = memref.alloc() : memref<128x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%67, %167, %169) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x3x3xf16>
    %170 = memref.alloc() : memref<4x128x28x28xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%22, %arg38, %162, %170, %arg127, %arg128) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x128x28x28xf16>, memref<128xf32>, memref<4x128x28x28xf16>, memref<4x128x28x28xf16>, memref<128xf32>, memref<128xf32>
    %171 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%170, %37, %171) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x128x28x28xf16>, memref<128x64x1x1xf16>, memref<4x64x56x56xf16>
    %172 = memref.alloc() : memref<128x64x1x1xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%67, %170, %172) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<0> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x128x28x28xf16>, memref<128x64x1x1xf16>
    %173 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PTXOp(%68, %171, %168, %173) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown123_kernel"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %174 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%23, %arg23, %173, %174, %arg118, %arg119) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %175 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%174, %36, %175) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %176 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%64, %174, %176) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %177 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PTXOp(%65, %175, %177) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown127_kernel"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %178 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%24, %arg18, %177, %178, %arg115, %arg116) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %179 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%178, %35, %179) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %180 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%61, %178, %180) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %181 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PTXOp(%62, %173, %179, %181) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown131_kernel"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %182 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%25, %arg13, %181, %182, %arg112, %arg113) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %183 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%182, %34, %183) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %184 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%58, %182, %184) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %185 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PTXOp(%59, %183, %185) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown135_kernel"} : memref<4x64x56x56xi1>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    %186 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%26, %arg8, %185, %186, %arg109, %arg110) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x56x56xf16>, memref<64xf32>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64xf32>, memref<64xf32>
    %187 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @ConvBackwardDataOpf16f16f16(%186, %33, %187) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<64x64x3x3xf16>, memref<4x64x56x56xf16>
    %188 = memref.alloc() : memref<64x64x3x3xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%27, %186, %188) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<1> : tensor<4xi64>, window_strides = dense<1> : tensor<2xi64>} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<64x64x3x3xf16>
    %189 = memref.alloc() : memref<4x64x56x56xf16>
    byre.compute @PTXOp(%181, %187, %189) {BlockSize.x = 128 : i32, GridSize.x = 6272 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown139_kernel"} : memref<4x64x56x56xf16>, memref<4x64x56x56xf16>, memref<4x64x56x56xf16>
    byre.compute @PoolMaxGradOpf16f16f16(%55, %189, %2) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : memref<4x64x112x112xf16>, memref<4x64x56x56xf16>, memref<4x64x112x112xf16>
    %190 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @PTXOp(%56, %2, %190) {BlockSize.x = 128 : i32, GridSize.x = 25088 : i32, arg_ranks = [4 : i32, 4 : i32, 4 : i32], kernel_name = "Unknown140_kernel"} : memref<4x64x112x112xi1>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>
    %191 = memref.alloc() : memref<4x64x112x112xf16>
    byre.compute @BatchNormGradOpf16f32f16f16f32f32(%29, %arg3, %190, %191, %arg106, %arg107) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : memref<4x64x112x112xf16>, memref<64xf32>, memref<4x64x112x112xf16>, memref<4x64x112x112xf16>, memref<64xf32>, memref<64xf32>
    %192 = memref.alloc() : memref<64x3x7x7xf16>
    byre.compute @ConvBackwardFilterOpf16f16f16(%30, %191, %192) {batch_group_count = 1 : i64, feature_group_count = 1 : i64, input_layout = "NCHW", kernel_layout = "NCHW", output_layout = "NCHW", padding = dense<3> : tensor<4xi64>, window_strides = dense<2> : tensor<2xi64>} : memref<4x3x224x224xf16>, memref<4x64x112x112xf16>, memref<64x3x7x7xf16>
    byre.compute @ReduceSumOpf32f32(%114, %1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<4x1000xf32>, memref<f32>
    byre.compute @PTXOp(%1, %arg104) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown143_kernel"} : memref<f32>, memref<f32>
    byre.compute @PTXOp(%192, %arg105) {BlockSize.x = 128 : i32, GridSize.x = 74 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown144_kernel"} : memref<64x3x7x7xf16>, memref<64x3x7x7xf32>
    byre.compute @PTXOp(%188, %arg108) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown145_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%184, %arg111) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown146_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%180, %arg114) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown147_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%176, %arg117) {BlockSize.x = 128 : i32, GridSize.x = 288 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown148_kernel"} : memref<64x64x3x3xf16>, memref<64x64x3x3xf32>
    byre.compute @PTXOp(%169, %arg120) {BlockSize.x = 128 : i32, GridSize.x = 576 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown149_kernel"} : memref<128x64x3x3xf16>, memref<128x64x3x3xf32>
    byre.compute @PTXOp(%165, %arg123) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown150_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%172, %arg126) {BlockSize.x = 128 : i32, GridSize.x = 64 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown151_kernel"} : memref<128x64x1x1xf16>, memref<128x64x1x1xf32>
    byre.compute @PTXOp(%161, %arg129) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown152_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%157, %arg132) {BlockSize.x = 128 : i32, GridSize.x = 1152 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown153_kernel"} : memref<128x128x3x3xf16>, memref<128x128x3x3xf32>
    byre.compute @PTXOp(%150, %arg135) {BlockSize.x = 128 : i32, GridSize.x = 2304 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown154_kernel"} : memref<256x128x3x3xf16>, memref<256x128x3x3xf32>
    byre.compute @PTXOp(%146, %arg138) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown155_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%153, %arg141) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown156_kernel"} : memref<256x128x1x1xf16>, memref<256x128x1x1xf32>
    byre.compute @PTXOp(%142, %arg144) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown157_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%138, %arg147) {BlockSize.x = 128 : i32, GridSize.x = 4608 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown158_kernel"} : memref<256x256x3x3xf16>, memref<256x256x3x3xf32>
    byre.compute @PTXOp(%131, %arg150) {BlockSize.x = 128 : i32, GridSize.x = 9216 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown159_kernel"} : memref<512x256x3x3xf16>, memref<512x256x3x3xf32>
    byre.compute @PTXOp(%127, %arg153) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown160_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%134, %arg156) {BlockSize.x = 128 : i32, GridSize.x = 1024 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown161_kernel"} : memref<512x256x1x1xf16>, memref<512x256x1x1xf32>
    byre.compute @PTXOp(%123, %arg159) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown162_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    byre.compute @PTXOp(%119, %arg162) {BlockSize.x = 128 : i32, GridSize.x = 18432 : i32, arg_ranks = [4 : i32, 4 : i32], kernel_name = "Unknown163_kernel"} : memref<512x512x3x3xf16>, memref<512x512x3x3xf32>
    %193 = memref.alloc() : memref<1000x512xf16>
    byre.compute @MatmulOpf16f16f16(%108, %113, %193) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<4x512xf16>, memref<4x1000xf16>, memref<1000x512xf16>
    byre.compute @PTXOp(%193, %arg165) {BlockSize.x = 128 : i32, GridSize.x = 4000 : i32, arg_ranks = [2 : i32, 2 : i32], kernel_name = "Unknown165_kernel"} : memref<1000x512xf16>, memref<1000x512xf32>
    byre.compute @ReduceSumOpf32f32(%115, %0) {dimensions = dense<0> : tensor<1xi64>} : memref<4x1000xf32>, memref<1000xf32>
    byre.compute @PTXOp(%0, %arg166) {BlockSize.x = 128 : i32, GridSize.x = 8 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown166_kernel"} : memref<1000xf32>, memref<1000xf32>
    return
  }
}

