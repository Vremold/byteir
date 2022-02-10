// RUN: byteir-translate -gen-ptx -dump-ptx %s | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown0_kernel
module attributes {byre.container_module, gpu.container_module}  {
  gpu.module @unified {
    llvm.func @Unknown0_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<i32>, %arg8: !llvm.ptr<i32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<i64>, %arg15: !llvm.ptr<i64>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr<i1>, %arg22: !llvm.ptr<i1>, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg7, %5[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.insertvalue %arg9, %7[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg10, %8[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg16, %11[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg17, %12[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg21, %14[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %arg22, %15[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg23, %16[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg24, %17[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.mlir.constant(0 : index) : i64
      %20 = llvm.mlir.constant(256 : index) : i64
      %21 = llvm.mlir.constant(30522 : i64) : i64
      %22 = llvm.mlir.constant(0 : i64) : i64
      %23 = llvm.mlir.constant(0.000000e+00 : f64) : f64
      %24 = llvm.mlir.constant(128 : index) : i64
      %25 = llvm.mlir.constant(-1 : index) : i64
      %26 = nvvm.read.ptx.sreg.ctaid.x : i32
      %27 = llvm.sext %26 : i32 to i64
      %28 = nvvm.read.ptx.sreg.tid.x : i32
      %29 = llvm.sext %28 : i32 to i64
      %30 = nvvm.read.ptx.sreg.ntid.x : i32
      %31 = llvm.sext %30 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %32 = llvm.mul %27, %31  : i64
      %33 = llvm.add %32, %29  : i64
      %34 = llvm.icmp "slt" %33, %20 : i64
      llvm.cond_br %34, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %35 = llvm.srem %33, %24  : i64
      %36 = llvm.icmp "slt" %35, %19 : i64
      %37 = llvm.add %35, %24  : i64
      %38 = llvm.select %36, %37, %35 : i1, i64
      %39 = llvm.icmp "slt" %33, %19 : i64
      %40 = llvm.sub %25, %33  : i64
      %41 = llvm.select %39, %40, %33 : i1, i64
      %42 = llvm.sdiv %41, %24  : i64
      %43 = llvm.sub %25, %42  : i64
      %44 = llvm.select %39, %43, %42 : i1, i64
      %45 = llvm.mul %44, %24  : i64
      %46 = llvm.add %45, %38  : i64
      %47 = llvm.getelementptr %arg1[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %48 = llvm.load %47 : !llvm.ptr<i64>
      %49 = llvm.trunc %48 : i64 to i32
      %50 = llvm.getelementptr %arg8[%46] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %49, %50 : !llvm.ptr<i32>
      %51 = llvm.add %48, %21  : i64
      %52 = llvm.icmp "slt" %48, %22 : i64
      %53 = llvm.select %52, %51, %48 : i1, i64
      %54 = llvm.getelementptr %arg15[%46] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %53, %54 : !llvm.ptr<i64>
      %55 = llvm.sitofp %48 : i64 to f64
      %56 = llvm.fcmp "une" %55, %23 : f64
      %57 = llvm.getelementptr %arg22[%46] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %56, %57 : !llvm.ptr<i1>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown1_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<i64>, %arg13: !llvm.ptr<i64>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<i1>, %arg20: !llvm.ptr<i1>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg12, %8[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg13, %9[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg14, %10[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg15, %11[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg19, %13[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg20, %14[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %arg21, %15[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg22, %16[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.mlir.constant(0 : index) : i64
      %19 = llvm.mlir.constant(256 : index) : i64
      %20 = llvm.mlir.constant(2 : i64) : i64
      %21 = llvm.mlir.constant(0 : i64) : i64
      %22 = llvm.mlir.constant(-1.000000e+00 : f64) : f64
      %23 = llvm.mlir.constant(128 : index) : i64
      %24 = llvm.mlir.constant(-1 : index) : i64
      %25 = nvvm.read.ptx.sreg.ctaid.x : i32
      %26 = llvm.sext %25 : i32 to i64
      %27 = nvvm.read.ptx.sreg.tid.x : i32
      %28 = llvm.sext %27 : i32 to i64
      %29 = nvvm.read.ptx.sreg.ntid.x : i32
      %30 = llvm.sext %29 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %31 = llvm.mul %26, %30  : i64
      %32 = llvm.add %31, %28  : i64
      %33 = llvm.icmp "slt" %32, %19 : i64
      llvm.cond_br %33, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %34 = llvm.srem %32, %23  : i64
      %35 = llvm.icmp "slt" %34, %18 : i64
      %36 = llvm.add %34, %23  : i64
      %37 = llvm.select %35, %36, %34 : i1, i64
      %38 = llvm.icmp "slt" %32, %18 : i64
      %39 = llvm.sub %24, %32  : i64
      %40 = llvm.select %38, %39, %32 : i1, i64
      %41 = llvm.sdiv %40, %23  : i64
      %42 = llvm.sub %24, %41  : i64
      %43 = llvm.select %38, %42, %41 : i1, i64
      %44 = llvm.getelementptr %arg1[%37] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %45 = llvm.load %44 : !llvm.ptr<i64>
      %46 = llvm.trunc %45 : i64 to i32
      %47 = llvm.mul %43, %23  : i64
      %48 = llvm.add %47, %37  : i64
      %49 = llvm.getelementptr %arg6[%48] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %46, %49 : !llvm.ptr<i32>
      %50 = llvm.add %45, %20  : i64
      %51 = llvm.icmp "slt" %45, %21 : i64
      %52 = llvm.select %51, %50, %45 : i1, i64
      %53 = llvm.getelementptr %arg13[%48] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %52, %53 : !llvm.ptr<i64>
      %54 = llvm.sitofp %45 : i64 to f64
      %55 = llvm.fcmp "une" %54, %22 : f64
      %56 = llvm.getelementptr %arg20[%48] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %55, %56 : !llvm.ptr<i1>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown2_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr<f32>, %arg19: !llvm.ptr<f32>, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %8 = llvm.insertvalue %arg10, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.insertvalue %arg11, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %10 = llvm.insertvalue %arg12, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %12 = llvm.insertvalue %arg13, %11[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg18, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg19, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg20, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.insertvalue %arg21, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.insertvalue %arg24, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %18 = llvm.insertvalue %arg22, %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.mlir.constant(0 : index) : i64
      %20 = llvm.mlir.constant(32768 : index) : i64
      %21 = llvm.mlir.constant(128 : index) : i64
      %22 = llvm.mlir.constant(-1 : index) : i64
      %23 = nvvm.read.ptx.sreg.ctaid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = nvvm.read.ptx.sreg.tid.x : i32
      %26 = llvm.sext %25 : i32 to i64
      %27 = nvvm.read.ptx.sreg.ntid.x : i32
      %28 = llvm.sext %27 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %29 = llvm.mul %24, %28  : i64
      %30 = llvm.add %29, %26  : i64
      %31 = llvm.icmp "slt" %30, %20 : i64
      llvm.cond_br %31, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %32 = llvm.srem %30, %21  : i64
      %33 = llvm.icmp "slt" %32, %19 : i64
      %34 = llvm.add %32, %21  : i64
      %35 = llvm.select %33, %34, %32 : i1, i64
      %36 = llvm.icmp "slt" %30, %19 : i64
      %37 = llvm.sub %22, %30  : i64
      %38 = llvm.select %36, %37, %30 : i1, i64
      %39 = llvm.sdiv %38, %21  : i64
      %40 = llvm.sub %22, %39  : i64
      %41 = llvm.select %36, %40, %39 : i1, i64
      %42 = llvm.srem %41, %21  : i64
      %43 = llvm.icmp "slt" %42, %19 : i64
      %44 = llvm.add %42, %21  : i64
      %45 = llvm.select %43, %44, %42 : i1, i64
      %46 = llvm.icmp "slt" %41, %19 : i64
      %47 = llvm.sub %22, %41  : i64
      %48 = llvm.select %46, %47, %41 : i1, i64
      %49 = llvm.sdiv %48, %21  : i64
      %50 = llvm.sub %22, %49  : i64
      %51 = llvm.select %46, %50, %49 : i1, i64
      %52 = llvm.mlir.constant(16384 : index) : i64
      %53 = llvm.mul %51, %52  : i64
      %54 = llvm.mul %45, %21  : i64
      %55 = llvm.add %53, %54  : i64
      %56 = llvm.add %55, %35  : i64
      %57 = llvm.getelementptr %arg1[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %58 = llvm.load %57 : !llvm.ptr<f32>
      %59 = llvm.getelementptr %arg10[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %60 = llvm.load %59 : !llvm.ptr<f32>
      %61 = llvm.fadd %58, %60  : f32
      %62 = llvm.getelementptr %arg19[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %61, %62 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown3_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<i64>, %arg11: !llvm.ptr<i64>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<i1>, %arg16: !llvm.ptr<i1>, %arg17: i64, %arg18: i64, %arg19: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %6 = llvm.insertvalue %arg10, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %arg11, %6[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.insertvalue %arg15, %8[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %10 = llvm.insertvalue %arg16, %9[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.mlir.constant(128 : index) : i64
      %12 = llvm.mlir.constant(512 : i64) : i64
      %13 = llvm.mlir.constant(0 : i64) : i64
      %14 = llvm.mlir.constant(-1.000000e+00 : f64) : f64
      %15 = nvvm.read.ptx.sreg.ctaid.x : i32
      %16 = llvm.sext %15 : i32 to i64
      %17 = nvvm.read.ptx.sreg.tid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.ntid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %21 = llvm.mul %16, %20  : i64
      %22 = llvm.add %21, %18  : i64
      %23 = llvm.icmp "slt" %22, %11 : i64
      llvm.cond_br %23, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %24 = llvm.getelementptr %arg1[%22] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %25 = llvm.load %24 : !llvm.ptr<i64>
      %26 = llvm.trunc %25 : i64 to i32
      %27 = llvm.getelementptr %arg6[%22] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %26, %27 : !llvm.ptr<i32>
      %28 = llvm.add %25, %12  : i64
      %29 = llvm.icmp "slt" %25, %13 : i64
      %30 = llvm.select %29, %28, %25 : i1, i64
      %31 = llvm.getelementptr %arg11[%22] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %30, %31 : !llvm.ptr<i64>
      %32 = llvm.sitofp %25 : i64 to f64
      %33 = llvm.fcmp "une" %32, %14 : f64
      %34 = llvm.getelementptr %arg16[%22] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %33, %34 : !llvm.ptr<i1>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown4_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.insertvalue %arg9, %7[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.insertvalue %arg10, %8[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %10 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %12 = llvm.insertvalue %arg16, %11[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg17, %12[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg20, %13[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg18, %14[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.mlir.constant(0 : index) : i64
      %17 = llvm.mlir.constant(7813632 : index) : i64
      %18 = llvm.mlir.constant(30522 : index) : i64
      %19 = llvm.mlir.constant(-1 : index) : i64
      %20 = llvm.mlir.constant(128 : index) : i64
      %21 = nvvm.read.ptx.sreg.ctaid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = nvvm.read.ptx.sreg.tid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = nvvm.read.ptx.sreg.ntid.x : i32
      %26 = llvm.sext %25 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %27 = llvm.mul %22, %26  : i64
      %28 = llvm.add %27, %24  : i64
      %29 = llvm.icmp "slt" %28, %17 : i64
      llvm.cond_br %29, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %30 = llvm.srem %28, %18  : i64
      %31 = llvm.icmp "slt" %30, %16 : i64
      %32 = llvm.add %30, %18  : i64
      %33 = llvm.select %31, %32, %30 : i1, i64
      %34 = llvm.icmp "slt" %28, %16 : i64
      %35 = llvm.sub %19, %28  : i64
      %36 = llvm.select %34, %35, %28 : i1, i64
      %37 = llvm.sdiv %36, %18  : i64
      %38 = llvm.sub %19, %37  : i64
      %39 = llvm.select %34, %38, %37 : i1, i64
      %40 = llvm.srem %39, %20  : i64
      %41 = llvm.icmp "slt" %40, %16 : i64
      %42 = llvm.add %40, %20  : i64
      %43 = llvm.select %41, %42, %40 : i1, i64
      %44 = llvm.icmp "slt" %39, %16 : i64
      %45 = llvm.sub %19, %39  : i64
      %46 = llvm.select %44, %45, %39 : i1, i64
      %47 = llvm.sdiv %46, %20  : i64
      %48 = llvm.sub %19, %47  : i64
      %49 = llvm.select %44, %48, %47 : i1, i64
      %50 = llvm.mlir.constant(3906816 : index) : i64
      %51 = llvm.mul %49, %50  : i64
      %52 = llvm.mul %43, %18  : i64
      %53 = llvm.add %51, %52  : i64
      %54 = llvm.add %53, %33  : i64
      %55 = llvm.getelementptr %arg1[%54] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %56 = llvm.load %55 : !llvm.ptr<f32>
      %57 = llvm.getelementptr %arg10[%33] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %58 = llvm.load %57 : !llvm.ptr<f32>
      %59 = llvm.fadd %56, %58  : f32
      %60 = llvm.getelementptr %arg15[%54] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %59, %60 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown5_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr<f32>, %arg19: !llvm.ptr<f32>, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: !llvm.ptr<f32>, %arg28: !llvm.ptr<f32>, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !llvm.ptr<f32>, %arg37: !llvm.ptr<f32>, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64, %arg44: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %8 = llvm.insertvalue %arg10, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.insertvalue %arg11, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %10 = llvm.insertvalue %arg12, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %12 = llvm.insertvalue %arg13, %11[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg18, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg19, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg20, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.insertvalue %arg21, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.insertvalue %arg24, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %18 = llvm.insertvalue %arg22, %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.insertvalue %arg27, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %20 = llvm.insertvalue %arg28, %19[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %21 = llvm.insertvalue %arg29, %20[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %22 = llvm.insertvalue %arg30, %21[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %23 = llvm.insertvalue %arg33, %22[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %24 = llvm.insertvalue %arg31, %23[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %25 = llvm.insertvalue %arg36, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %26 = llvm.insertvalue %arg37, %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %27 = llvm.insertvalue %arg38, %26[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %28 = llvm.insertvalue %arg39, %27[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %29 = llvm.insertvalue %arg42, %28[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %30 = llvm.insertvalue %arg40, %29[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %31 = llvm.mlir.constant(0 : index) : i64
      %32 = llvm.mlir.constant(32768 : index) : i64
      %33 = llvm.mlir.constant(128 : index) : i64
      %34 = llvm.mlir.constant(-1 : index) : i64
      %35 = nvvm.read.ptx.sreg.ctaid.x : i32
      %36 = llvm.sext %35 : i32 to i64
      %37 = nvvm.read.ptx.sreg.tid.x : i32
      %38 = llvm.sext %37 : i32 to i64
      %39 = nvvm.read.ptx.sreg.ntid.x : i32
      %40 = llvm.sext %39 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %41 = llvm.mul %36, %40  : i64
      %42 = llvm.add %41, %38  : i64
      %43 = llvm.icmp "slt" %42, %32 : i64
      llvm.cond_br %43, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %44 = llvm.srem %42, %33  : i64
      %45 = llvm.icmp "slt" %44, %31 : i64
      %46 = llvm.add %44, %33  : i64
      %47 = llvm.select %45, %46, %44 : i1, i64
      %48 = llvm.icmp "slt" %42, %31 : i64
      %49 = llvm.sub %34, %42  : i64
      %50 = llvm.select %48, %49, %42 : i1, i64
      %51 = llvm.sdiv %50, %33  : i64
      %52 = llvm.sub %34, %51  : i64
      %53 = llvm.select %48, %52, %51 : i1, i64
      %54 = llvm.srem %53, %33  : i64
      %55 = llvm.icmp "slt" %54, %31 : i64
      %56 = llvm.add %54, %33  : i64
      %57 = llvm.select %55, %56, %54 : i1, i64
      %58 = llvm.icmp "slt" %53, %31 : i64
      %59 = llvm.sub %34, %53  : i64
      %60 = llvm.select %58, %59, %53 : i1, i64
      %61 = llvm.sdiv %60, %33  : i64
      %62 = llvm.sub %34, %61  : i64
      %63 = llvm.select %58, %62, %61 : i1, i64
      %64 = llvm.mlir.constant(16384 : index) : i64
      %65 = llvm.mul %63, %64  : i64
      %66 = llvm.mul %57, %33  : i64
      %67 = llvm.add %65, %66  : i64
      %68 = llvm.add %67, %47  : i64
      %69 = llvm.getelementptr %arg1[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %70 = llvm.load %69 : !llvm.ptr<f32>
      %71 = llvm.getelementptr %arg10[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %72 = llvm.load %71 : !llvm.ptr<f32>
      %73 = llvm.getelementptr %arg19[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %74 = llvm.load %73 : !llvm.ptr<f32>
      %75 = llvm.getelementptr %arg28[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %76 = llvm.load %75 : !llvm.ptr<f32>
      %77 = llvm.fadd %70, %72  : f32
      %78 = llvm.fadd %77, %74  : f32
      %79 = llvm.fadd %78, %76  : f32
      %80 = llvm.getelementptr %arg37[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %79, %80 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown6_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr<f32>, %arg19: !llvm.ptr<f32>, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: !llvm.ptr<f32>, %arg28: !llvm.ptr<f32>, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: !llvm.ptr<f32>, %arg37: !llvm.ptr<f32>, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64, %arg44: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %8 = llvm.insertvalue %arg10, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.insertvalue %arg11, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %10 = llvm.insertvalue %arg12, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %12 = llvm.insertvalue %arg13, %11[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg18, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg19, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg20, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.insertvalue %arg21, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.insertvalue %arg24, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %18 = llvm.insertvalue %arg22, %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.insertvalue %arg27, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %20 = llvm.insertvalue %arg28, %19[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %21 = llvm.insertvalue %arg29, %20[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %22 = llvm.insertvalue %arg30, %21[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %23 = llvm.insertvalue %arg33, %22[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %24 = llvm.insertvalue %arg31, %23[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %25 = llvm.insertvalue %arg36, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %26 = llvm.insertvalue %arg37, %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %27 = llvm.insertvalue %arg38, %26[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %28 = llvm.insertvalue %arg39, %27[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %29 = llvm.insertvalue %arg42, %28[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %30 = llvm.insertvalue %arg40, %29[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %31 = llvm.mlir.constant(0 : index) : i64
      %32 = llvm.mlir.constant(32768 : index) : i64
      %33 = llvm.mlir.constant(128 : index) : i64
      %34 = llvm.mlir.constant(-1 : index) : i64
      %35 = nvvm.read.ptx.sreg.ctaid.x : i32
      %36 = llvm.sext %35 : i32 to i64
      %37 = nvvm.read.ptx.sreg.tid.x : i32
      %38 = llvm.sext %37 : i32 to i64
      %39 = nvvm.read.ptx.sreg.ntid.x : i32
      %40 = llvm.sext %39 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %41 = llvm.mul %36, %40  : i64
      %42 = llvm.add %41, %38  : i64
      %43 = llvm.icmp "slt" %42, %32 : i64
      llvm.cond_br %43, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %44 = llvm.srem %42, %33  : i64
      %45 = llvm.icmp "slt" %44, %31 : i64
      %46 = llvm.add %44, %33  : i64
      %47 = llvm.select %45, %46, %44 : i1, i64
      %48 = llvm.icmp "slt" %42, %31 : i64
      %49 = llvm.sub %34, %42  : i64
      %50 = llvm.select %48, %49, %42 : i1, i64
      %51 = llvm.sdiv %50, %33  : i64
      %52 = llvm.sub %34, %51  : i64
      %53 = llvm.select %48, %52, %51 : i1, i64
      %54 = llvm.srem %53, %33  : i64
      %55 = llvm.icmp "slt" %54, %31 : i64
      %56 = llvm.add %54, %33  : i64
      %57 = llvm.select %55, %56, %54 : i1, i64
      %58 = llvm.icmp "slt" %53, %31 : i64
      %59 = llvm.sub %34, %53  : i64
      %60 = llvm.select %58, %59, %53 : i1, i64
      %61 = llvm.sdiv %60, %33  : i64
      %62 = llvm.sub %34, %61  : i64
      %63 = llvm.select %58, %62, %61 : i1, i64
      %64 = llvm.mlir.constant(16384 : index) : i64
      %65 = llvm.mul %63, %64  : i64
      %66 = llvm.mul %57, %33  : i64
      %67 = llvm.add %65, %66  : i64
      %68 = llvm.add %67, %47  : i64
      %69 = llvm.getelementptr %arg1[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %70 = llvm.load %69 : !llvm.ptr<f32>
      %71 = llvm.getelementptr %arg10[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %72 = llvm.load %71 : !llvm.ptr<f32>
      %73 = llvm.getelementptr %arg19[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %74 = llvm.load %73 : !llvm.ptr<f32>
      %75 = llvm.getelementptr %arg28[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %76 = llvm.load %75 : !llvm.ptr<f32>
      %77 = llvm.fadd %70, %72  : f32
      %78 = llvm.fadd %77, %74  : f32
      %79 = llvm.fadd %78, %76  : f32
      %80 = llvm.getelementptr %arg37[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %79, %80 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown7_kernel(%arg0: !llvm.ptr<i1>, %arg1: !llvm.ptr<i1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr<f32>, %arg17: !llvm.ptr<f32>, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr<i1>, %arg26: !llvm.ptr<i1>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: !llvm.ptr<f32>, %arg33: !llvm.ptr<f32>, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %6 = llvm.insertvalue %arg7, %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %8 = llvm.insertvalue %arg9, %7[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %9 = llvm.insertvalue %arg10, %8[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %10 = llvm.insertvalue %arg13, %9[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %11 = llvm.insertvalue %arg11, %10[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %12 = llvm.insertvalue %arg16, %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %13 = llvm.insertvalue %arg17, %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %14 = llvm.insertvalue %arg18, %13[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %15 = llvm.insertvalue %arg19, %14[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %16 = llvm.insertvalue %arg22, %15[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %17 = llvm.insertvalue %arg20, %16[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %18 = llvm.insertvalue %arg25, %0[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg26, %18[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg27, %19[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg28, %20[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg32, %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %23 = llvm.insertvalue %arg33, %22[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %24 = llvm.insertvalue %arg34, %23[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %25 = llvm.insertvalue %arg35, %24[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %26 = llvm.insertvalue %arg38, %25[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %27 = llvm.insertvalue %arg36, %26[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %28 = llvm.mlir.constant(0 : index) : i64
      %29 = llvm.mlir.constant(32768 : index) : i64
      %30 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %31 = llvm.mlir.constant(128 : index) : i64
      %32 = llvm.mlir.constant(-1 : index) : i64
      %33 = nvvm.read.ptx.sreg.ctaid.x : i32
      %34 = llvm.sext %33 : i32 to i64
      %35 = nvvm.read.ptx.sreg.tid.x : i32
      %36 = llvm.sext %35 : i32 to i64
      %37 = nvvm.read.ptx.sreg.ntid.x : i32
      %38 = llvm.sext %37 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %39 = llvm.mul %34, %38  : i64
      %40 = llvm.add %39, %36  : i64
      %41 = llvm.icmp "slt" %40, %29 : i64
      llvm.cond_br %41, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %42 = llvm.srem %40, %31  : i64
      %43 = llvm.icmp "slt" %42, %28 : i64
      %44 = llvm.add %42, %31  : i64
      %45 = llvm.select %43, %44, %42 : i1, i64
      %46 = llvm.icmp "slt" %40, %28 : i64
      %47 = llvm.sub %32, %40  : i64
      %48 = llvm.select %46, %47, %40 : i1, i64
      %49 = llvm.sdiv %48, %31  : i64
      %50 = llvm.sub %32, %49  : i64
      %51 = llvm.select %46, %50, %49 : i1, i64
      %52 = llvm.srem %51, %31  : i64
      %53 = llvm.icmp "slt" %52, %28 : i64
      %54 = llvm.add %52, %31  : i64
      %55 = llvm.select %53, %54, %52 : i1, i64
      %56 = llvm.icmp "slt" %51, %28 : i64
      %57 = llvm.sub %32, %51  : i64
      %58 = llvm.select %56, %57, %51 : i1, i64
      %59 = llvm.sdiv %58, %31  : i64
      %60 = llvm.sub %32, %59  : i64
      %61 = llvm.select %56, %60, %59 : i1, i64
      %62 = llvm.mul %61, %31  : i64
      %63 = llvm.add %62, %55  : i64
      %64 = llvm.getelementptr %arg1[%63] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %65 = llvm.load %64 : !llvm.ptr<i1>
      %66 = llvm.mlir.constant(16384 : index) : i64
      %67 = llvm.mul %61, %66  : i64
      %68 = llvm.mul %55, %31  : i64
      %69 = llvm.add %67, %68  : i64
      %70 = llvm.add %69, %45  : i64
      %71 = llvm.getelementptr %arg8[%70] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %72 = llvm.load %71 : !llvm.ptr<f32>
      %73 = llvm.select %65, %72, %30 : i1, f32
      %74 = llvm.getelementptr %arg17[%70] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %73, %74 : !llvm.ptr<f32>
      %75 = llvm.getelementptr %arg26[%63] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %76 = llvm.load %75 : !llvm.ptr<i1>
      %77 = llvm.select %76, %72, %30 : i1, f32
      %78 = llvm.getelementptr %arg33[%70] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %77, %78 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown8_kernel(%arg0: !llvm.ptr<i1>, %arg1: !llvm.ptr<i1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<f32>, %arg13: !llvm.ptr<f32>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.insertvalue %arg12, %3[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg13, %8[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg14, %9[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg15, %10[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.mlir.constant(0 : index) : i64
      %13 = llvm.mlir.constant(16384 : index) : i64
      %14 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %15 = llvm.mlir.constant(128 : index) : i64
      %16 = llvm.mlir.constant(-1 : index) : i64
      %17 = nvvm.read.ptx.sreg.ctaid.x : i32
      %18 = llvm.sext %17 : i32 to i64
      %19 = nvvm.read.ptx.sreg.tid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = nvvm.read.ptx.sreg.ntid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %23 = llvm.mul %18, %22  : i64
      %24 = llvm.add %23, %20  : i64
      %25 = llvm.icmp "slt" %24, %13 : i64
      llvm.cond_br %25, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %26 = llvm.srem %24, %15  : i64
      %27 = llvm.icmp "slt" %26, %12 : i64
      %28 = llvm.add %26, %15  : i64
      %29 = llvm.select %27, %28, %26 : i1, i64
      %30 = llvm.icmp "slt" %24, %12 : i64
      %31 = llvm.sub %16, %24  : i64
      %32 = llvm.select %30, %31, %24 : i1, i64
      %33 = llvm.sdiv %32, %15  : i64
      %34 = llvm.sub %16, %33  : i64
      %35 = llvm.select %30, %34, %33 : i1, i64
      %36 = llvm.getelementptr %arg1[%35] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %37 = llvm.load %36 : !llvm.ptr<i1>
      %38 = llvm.mul %35, %15  : i64
      %39 = llvm.add %38, %29  : i64
      %40 = llvm.getelementptr %arg6[%39] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %41 = llvm.load %40 : !llvm.ptr<f32>
      %42 = llvm.select %37, %41, %14 : i1, f32
      %43 = llvm.getelementptr %arg13[%39] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %42, %43 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
  }
}

