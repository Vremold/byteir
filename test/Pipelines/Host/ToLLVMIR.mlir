// RUN: byteir-translate %s -mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @Unknown1
// CHECK-LABEL: define void @Unknown0
module attributes {byre.container_module, llvm.data_layout = ""} {
  llvm.mlir.global private constant @__constant_1x128xi32(dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000"> : tensor<1x128xi32>) : !llvm.array<1 x array<128 x i32>>
  llvm.func @Unknown1(%arg0: !llvm.ptr<i32>, %arg1: !llvm.ptr<i32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<i32>, %arg13: !llvm.ptr<i32>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<i32>, %arg20: !llvm.ptr<i32>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(-1 : index) : i64
    %4 = llvm.mlir.addressof @__constant_1x128xi32 : !llvm.ptr<array<1 x array<128 x i32>>>
    %5 = llvm.getelementptr %4[%0, %0, %0] : (!llvm.ptr<array<1 x array<128 x i32>>>, i64, i64, i64) -> !llvm.ptr<i32>
    llvm.br ^bb1(%0 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
    %7 = llvm.icmp "slt" %6, %1 : i64
    llvm.cond_br %7, ^bb2, ^bb3(%0 : i64)
  ^bb2:  // pred: ^bb1
    %8 = llvm.icmp "slt" %6, %0 : i64
    %9 = llvm.add %6, %1  : i64
    %10 = llvm.select %8, %9, %6 : i1, i64
    %11 = llvm.sub %3, %6  : i64
    %12 = llvm.select %8, %11, %6 : i1, i64
    %13 = llvm.sdiv %12, %1  : i64
    %14 = llvm.sub %3, %13  : i64
    %15 = llvm.select %8, %14, %13 : i1, i64
    %16 = llvm.mul %15, %1  : i64
    %17 = llvm.add %16, %10  : i64
    %18 = llvm.getelementptr %5[%17] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %19 = llvm.load %18 : !llvm.ptr<i32>
    %20 = llvm.getelementptr %arg1[%15] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %21 = llvm.load %20 : !llvm.ptr<i32>
    %22 = llvm.icmp "slt" %19, %21 : i32
    %23 = llvm.zext %22 : i1 to i32
    %24 = llvm.mul %15, %1  : i64
    %25 = llvm.add %24, %10  : i64
    %26 = llvm.getelementptr %arg13[%25] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %23, %26 : !llvm.ptr<i32>
    %27 = llvm.add %6, %2  : i64
    llvm.br ^bb1(%27 : i64)
  ^bb3(%28: i64):  // 2 preds: ^bb1, ^bb4
    %29 = llvm.icmp "slt" %28, %1 : i64
    llvm.cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %30 = llvm.icmp "slt" %28, %0 : i64
    %31 = llvm.add %28, %1  : i64
    %32 = llvm.select %30, %31, %28 : i1, i64
    %33 = llvm.sub %3, %28  : i64
    %34 = llvm.select %30, %33, %28 : i1, i64
    %35 = llvm.sdiv %34, %1  : i64
    %36 = llvm.sub %3, %35  : i64
    %37 = llvm.select %30, %36, %35 : i1, i64
    %38 = llvm.mul %37, %1  : i64
    %39 = llvm.add %38, %32  : i64
    %40 = llvm.getelementptr %arg13[%39] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %41 = llvm.load %40 : !llvm.ptr<i32>
    %42 = llvm.mul %37, %1  : i64
    %43 = llvm.add %42, %32  : i64
    %44 = llvm.getelementptr %arg6[%43] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    %45 = llvm.load %44 : !llvm.ptr<i32>
    %46 = llvm.mul %41, %45  : i32
    %47 = llvm.mul %37, %1  : i64
    %48 = llvm.add %47, %32  : i64
    %49 = llvm.getelementptr %arg20[%48] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
    llvm.store %46, %49 : !llvm.ptr<i32>
    %50 = llvm.add %28, %2  : i64
    llvm.br ^bb3(%50 : i64)
  ^bb5:  // pred: ^bb3
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown1(%arg0: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %6[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %6[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %14[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.extractvalue %14[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.extractvalue %14[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>>
    %23 = llvm.extractvalue %22[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.extractvalue %22[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.extractvalue %22[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.extractvalue %22[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.extractvalue %22[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.extractvalue %22[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.extractvalue %22[4, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @Unknown1(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %12, %13, %15, %16, %17, %18, %19, %20, %21, %23, %24, %25, %26, %27, %28, %29) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64, i64, i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64, i64, i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @Unknown0(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i64>, %arg6: !llvm.ptr<i64>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<i64>, %arg11: !llvm.ptr<i64>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<i32>, %arg16: !llvm.ptr<i32>, %arg17: i64, %arg18: i64, %arg19: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg11 : !llvm.ptr<i64>
    %1 = llvm.load %arg1 : !llvm.ptr<i64>
    %2 = llvm.load %arg6 : !llvm.ptr<i64>
    %3 = llvm.add %1, %2  : i64
    %4 = llvm.add %0, %3  : i64
    %5 = llvm.trunc %4 : i64 to i32
    llvm.store %5, %arg16 : !llvm.ptr<i32>
    llvm.return
  }
  llvm.func @_mlir_ciface_Unknown0(%arg0: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %18 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>>
    %19 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    %23 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.call @Unknown0(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %13, %14, %15, %16, %17, %19, %20, %21, %22, %23) : (!llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i64, i64, i64) -> ()
    llvm.return
  }
}

