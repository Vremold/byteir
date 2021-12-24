// RUN: byteir-translate -gen-ptx -dump-ptx %s | FileCheck %s

// CHECK-LABEL: .visible .entry Unknown0_kernel

module attributes {byre.container_module, gpu.container_module}  {
  gpu.module @unified {
    llvm.func @Unknown0_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<i32>, %arg8: !llvm.ptr<i32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<i64>, %arg15: !llvm.ptr<i64>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr<i64>, %arg22: !llvm.ptr<i64>, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: !llvm.ptr<i64>, %arg29: !llvm.ptr<i64>, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: !llvm.ptr<f64>, %arg36: !llvm.ptr<f64>, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: i64, %arg42: !llvm.ptr<i1>, %arg43: !llvm.ptr<i1>, %arg44: i64, %arg45: i64, %arg46: i64, %arg47: i64, %arg48: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %14 = llvm.insertvalue %arg21, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg22, %14[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %arg23, %15[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg24, %16[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg28, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg29, %18[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg30, %19[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg31, %20[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg35, %22[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %24 = llvm.insertvalue %arg36, %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %25 = llvm.insertvalue %arg37, %24[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %26 = llvm.insertvalue %arg38, %25[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %27 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %28 = llvm.insertvalue %arg42, %27[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %29 = llvm.insertvalue %arg43, %28[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %30 = llvm.insertvalue %arg44, %29[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %31 = llvm.insertvalue %arg45, %30[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %32 = llvm.mlir.constant(0 : index) : i64
      %33 = llvm.mlir.constant(256 : index) : i64
      %34 = llvm.mlir.constant(128 : index) : i64
      %35 = llvm.mlir.constant(-1 : index) : i64
      %36 = nvvm.read.ptx.sreg.ctaid.x : i32
      %37 = llvm.sext %36 : i32 to i64
      %38 = nvvm.read.ptx.sreg.tid.x : i32
      %39 = llvm.sext %38 : i32 to i64
      %40 = nvvm.read.ptx.sreg.ntid.x : i32
      %41 = llvm.sext %40 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %42 = llvm.mul %37, %41  : i64
      %43 = llvm.add %42, %39  : i64
      %44 = llvm.icmp "slt" %43, %33 : i64
      llvm.cond_br %44, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %45 = llvm.srem %43, %34  : i64
      %46 = llvm.icmp "slt" %45, %32 : i64
      %47 = llvm.add %45, %34  : i64
      %48 = llvm.select %46, %47, %45 : i1, i64
      %49 = llvm.icmp "slt" %43, %32 : i64
      %50 = llvm.sub %35, %43  : i64
      %51 = llvm.select %49, %50, %43 : i1, i64
      %52 = llvm.sdiv %51, %34  : i64
      %53 = llvm.sub %35, %52  : i64
      %54 = llvm.select %49, %53, %52 : i1, i64
      %55 = llvm.mul %54, %34  : i64
      %56 = llvm.add %55, %48  : i64
      %57 = llvm.getelementptr %arg1[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %58 = llvm.load %57 : !llvm.ptr<i64>
      %59 = llvm.trunc %58 : i64 to i32
      %60 = llvm.getelementptr %arg8[%56] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %59, %60 : !llvm.ptr<i32>
      %61 = llvm.getelementptr %arg15[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %62 = llvm.load %61 : !llvm.ptr<i64>
      %63 = llvm.getelementptr %arg22[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %64 = llvm.load %63 : !llvm.ptr<i64>
      %65 = llvm.add %58, %64  : i64
      %66 = llvm.icmp "slt" %58, %62 : i64
      %67 = llvm.select %66, %65, %58 : i1, i64
      %68 = llvm.getelementptr %arg29[%56] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %67, %68 : !llvm.ptr<i64>
      %69 = llvm.getelementptr %arg36[%56] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %70 = llvm.load %69 : !llvm.ptr<f64>
      %71 = llvm.sitofp %58 : i64 to f64
      %72 = llvm.fcmp "une" %71, %70 : f64
      %73 = llvm.getelementptr %arg43[%56] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %72, %73 : !llvm.ptr<i1>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown1_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<i64>, %arg13: !llvm.ptr<i64>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<i64>, %arg20: !llvm.ptr<i64>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr<i64>, %arg27: !llvm.ptr<i64>, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: !llvm.ptr<f64>, %arg34: !llvm.ptr<f64>, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: !llvm.ptr<i1>, %arg41: !llvm.ptr<i1>, %arg42: i64, %arg43: i64, %arg44: i64, %arg45: i64, %arg46: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %13 = llvm.insertvalue %arg19, %8[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg20, %13[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg21, %14[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %arg22, %15[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg26, %8[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg27, %17[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg28, %18[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg29, %19[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg33, %21[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg34, %22[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %24 = llvm.insertvalue %arg35, %23[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %25 = llvm.insertvalue %arg36, %24[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %26 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %27 = llvm.insertvalue %arg40, %26[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %28 = llvm.insertvalue %arg41, %27[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %29 = llvm.insertvalue %arg42, %28[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %30 = llvm.insertvalue %arg43, %29[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %31 = llvm.mlir.constant(0 : index) : i64
      %32 = llvm.mlir.constant(256 : index) : i64
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
      %54 = llvm.getelementptr %arg1[%47] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %55 = llvm.load %54 : !llvm.ptr<i64>
      %56 = llvm.trunc %55 : i64 to i32
      %57 = llvm.mul %53, %33  : i64
      %58 = llvm.add %57, %47  : i64
      %59 = llvm.getelementptr %arg6[%58] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %56, %59 : !llvm.ptr<i32>
      %60 = llvm.getelementptr %arg13[%58] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %61 = llvm.load %60 : !llvm.ptr<i64>
      %62 = llvm.getelementptr %arg20[%58] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %63 = llvm.load %62 : !llvm.ptr<i64>
      %64 = llvm.add %55, %63  : i64
      %65 = llvm.icmp "slt" %55, %61 : i64
      %66 = llvm.select %65, %64, %55 : i1, i64
      %67 = llvm.getelementptr %arg27[%58] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %66, %67 : !llvm.ptr<i64>
      %68 = llvm.getelementptr %arg34[%58] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %69 = llvm.load %68 : !llvm.ptr<f64>
      %70 = llvm.sitofp %55 : i64 to f64
      %71 = llvm.fcmp "une" %70, %69 : f64
      %72 = llvm.getelementptr %arg41[%58] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %71, %72 : !llvm.ptr<i1>
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
    llvm.func @Unknown3_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i32>, %arg6: !llvm.ptr<i32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<i64>, %arg11: !llvm.ptr<i64>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<i64>, %arg16: !llvm.ptr<i64>, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: !llvm.ptr<i64>, %arg21: !llvm.ptr<i64>, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr<f64>, %arg26: !llvm.ptr<f64>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: !llvm.ptr<i1>, %arg31: !llvm.ptr<i1>, %arg32: i64, %arg33: i64, %arg34: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
      %6 = llvm.insertvalue %arg10, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %arg11, %6[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %8 = llvm.insertvalue %arg15, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %9 = llvm.insertvalue %arg16, %8[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %10 = llvm.insertvalue %arg20, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %arg21, %10[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %12 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %13 = llvm.insertvalue %arg25, %12[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %14 = llvm.insertvalue %arg26, %13[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %15 = llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %16 = llvm.insertvalue %arg30, %15[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %17 = llvm.insertvalue %arg31, %16[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
      %18 = llvm.mlir.constant(128 : index) : i64
      %19 = nvvm.read.ptx.sreg.ctaid.x : i32
      %20 = llvm.sext %19 : i32 to i64
      %21 = nvvm.read.ptx.sreg.tid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = nvvm.read.ptx.sreg.ntid.x : i32
      %24 = llvm.sext %23 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %25 = llvm.mul %20, %24  : i64
      %26 = llvm.add %25, %22  : i64
      %27 = llvm.icmp "slt" %26, %18 : i64
      llvm.cond_br %27, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %28 = llvm.getelementptr %arg1[%26] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %29 = llvm.load %28 : !llvm.ptr<i64>
      %30 = llvm.trunc %29 : i64 to i32
      %31 = llvm.getelementptr %arg6[%26] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
      llvm.store %30, %31 : !llvm.ptr<i32>
      %32 = llvm.getelementptr %arg11[%26] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %33 = llvm.load %32 : !llvm.ptr<i64>
      %34 = llvm.getelementptr %arg16[%26] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %35 = llvm.load %34 : !llvm.ptr<i64>
      %36 = llvm.add %29, %35  : i64
      %37 = llvm.icmp "slt" %29, %33 : i64
      %38 = llvm.select %37, %36, %29 : i1, i64
      %39 = llvm.getelementptr %arg21[%26] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      llvm.store %38, %39 : !llvm.ptr<i64>
      %40 = llvm.getelementptr %arg26[%26] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %41 = llvm.load %40 : !llvm.ptr<f64>
      %42 = llvm.sitofp %29 : i64 to f64
      %43 = llvm.fcmp "une" %42, %41 : f64
      %44 = llvm.getelementptr %arg31[%26] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      llvm.store %43, %44 : !llvm.ptr<i1>
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
    llvm.func @Unknown7_kernel(%arg0: !llvm.ptr<i1>, %arg1: !llvm.ptr<i1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr<f32>, %arg17: !llvm.ptr<f32>, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: !llvm.ptr<f32>, %arg26: !llvm.ptr<f32>, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: !llvm.ptr<i1>, %arg35: !llvm.ptr<i1>, %arg36: i64, %arg37: i64, %arg38: i64, %arg39: i64, %arg40: i64, %arg41: !llvm.ptr<f32>, %arg42: !llvm.ptr<f32>, %arg43: i64, %arg44: i64, %arg45: i64, %arg46: i64, %arg47: i64, %arg48: i64, %arg49: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %18 = llvm.insertvalue %arg25, %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %19 = llvm.insertvalue %arg26, %18[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %20 = llvm.insertvalue %arg27, %19[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %21 = llvm.insertvalue %arg28, %20[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %22 = llvm.insertvalue %arg31, %21[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %23 = llvm.insertvalue %arg29, %22[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %24 = llvm.insertvalue %arg34, %0[0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %25 = llvm.insertvalue %arg35, %24[1] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %26 = llvm.insertvalue %arg36, %25[2] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %27 = llvm.insertvalue %arg37, %26[3, 0] : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<2 x i64>, array<2 x i64>)>
      %28 = llvm.insertvalue %arg41, %5[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %29 = llvm.insertvalue %arg42, %28[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %30 = llvm.insertvalue %arg43, %29[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %31 = llvm.insertvalue %arg44, %30[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %32 = llvm.insertvalue %arg47, %31[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %33 = llvm.insertvalue %arg45, %32[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
      %34 = llvm.mlir.constant(0 : index) : i64
      %35 = llvm.mlir.constant(32768 : index) : i64
      %36 = llvm.mlir.constant(128 : index) : i64
      %37 = llvm.mlir.constant(-1 : index) : i64
      %38 = nvvm.read.ptx.sreg.ctaid.x : i32
      %39 = llvm.sext %38 : i32 to i64
      %40 = nvvm.read.ptx.sreg.tid.x : i32
      %41 = llvm.sext %40 : i32 to i64
      %42 = nvvm.read.ptx.sreg.ntid.x : i32
      %43 = llvm.sext %42 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %44 = llvm.mul %39, %43  : i64
      %45 = llvm.add %44, %41  : i64
      %46 = llvm.icmp "slt" %45, %35 : i64
      llvm.cond_br %46, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %47 = llvm.srem %45, %36  : i64
      %48 = llvm.icmp "slt" %47, %34 : i64
      %49 = llvm.add %47, %36  : i64
      %50 = llvm.select %48, %49, %47 : i1, i64
      %51 = llvm.icmp "slt" %45, %34 : i64
      %52 = llvm.sub %37, %45  : i64
      %53 = llvm.select %51, %52, %45 : i1, i64
      %54 = llvm.sdiv %53, %36  : i64
      %55 = llvm.sub %37, %54  : i64
      %56 = llvm.select %51, %55, %54 : i1, i64
      %57 = llvm.srem %56, %36  : i64
      %58 = llvm.icmp "slt" %57, %34 : i64
      %59 = llvm.add %57, %36  : i64
      %60 = llvm.select %58, %59, %57 : i1, i64
      %61 = llvm.icmp "slt" %56, %34 : i64
      %62 = llvm.sub %37, %56  : i64
      %63 = llvm.select %61, %62, %56 : i1, i64
      %64 = llvm.sdiv %63, %36  : i64
      %65 = llvm.sub %37, %64  : i64
      %66 = llvm.select %61, %65, %64 : i1, i64
      %67 = llvm.mul %66, %36  : i64
      %68 = llvm.add %67, %60  : i64
      %69 = llvm.getelementptr %arg1[%68] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %70 = llvm.load %69 : !llvm.ptr<i1>
      %71 = llvm.mlir.constant(16384 : index) : i64
      %72 = llvm.mul %66, %71  : i64
      %73 = llvm.mul %60, %36  : i64
      %74 = llvm.add %72, %73  : i64
      %75 = llvm.add %74, %50  : i64
      %76 = llvm.getelementptr %arg8[%75] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %77 = llvm.load %76 : !llvm.ptr<f32>
      %78 = llvm.getelementptr %arg17[%75] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %79 = llvm.load %78 : !llvm.ptr<f32>
      %80 = llvm.select %70, %77, %79 : i1, f32
      %81 = llvm.getelementptr %arg26[%75] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %80, %81 : !llvm.ptr<f32>
      %82 = llvm.getelementptr %arg35[%68] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %83 = llvm.load %82 : !llvm.ptr<i1>
      %84 = llvm.select %83, %77, %79 : i1, f32
      %85 = llvm.getelementptr %arg42[%75] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %84, %85 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
    llvm.func @Unknown8_kernel(%arg0: !llvm.ptr<i1>, %arg1: !llvm.ptr<i1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr<f32>, %arg13: !llvm.ptr<f32>, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: !llvm.ptr<f32>, %arg20: !llvm.ptr<f32>, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %12 = llvm.insertvalue %arg19, %3[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg20, %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg21, %13[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg22, %14[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.mlir.constant(0 : index) : i64
      %17 = llvm.mlir.constant(16384 : index) : i64
      %18 = llvm.mlir.constant(128 : index) : i64
      %19 = llvm.mlir.constant(-1 : index) : i64
      %20 = nvvm.read.ptx.sreg.ctaid.x : i32
      %21 = llvm.sext %20 : i32 to i64
      %22 = nvvm.read.ptx.sreg.tid.x : i32
      %23 = llvm.sext %22 : i32 to i64
      %24 = nvvm.read.ptx.sreg.ntid.x : i32
      %25 = llvm.sext %24 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %26 = llvm.mul %21, %25  : i64
      %27 = llvm.add %26, %23  : i64
      %28 = llvm.icmp "slt" %27, %17 : i64
      llvm.cond_br %28, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %29 = llvm.srem %27, %18  : i64
      %30 = llvm.icmp "slt" %29, %16 : i64
      %31 = llvm.add %29, %18  : i64
      %32 = llvm.select %30, %31, %29 : i1, i64
      %33 = llvm.icmp "slt" %27, %16 : i64
      %34 = llvm.sub %19, %27  : i64
      %35 = llvm.select %33, %34, %27 : i1, i64
      %36 = llvm.sdiv %35, %18  : i64
      %37 = llvm.sub %19, %36  : i64
      %38 = llvm.select %33, %37, %36 : i1, i64
      %39 = llvm.getelementptr %arg1[%38] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
      %40 = llvm.load %39 : !llvm.ptr<i1>
      %41 = llvm.mul %38, %18  : i64
      %42 = llvm.add %41, %32  : i64
      %43 = llvm.getelementptr %arg6[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %44 = llvm.load %43 : !llvm.ptr<f32>
      %45 = llvm.getelementptr %arg13[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %46 = llvm.load %45 : !llvm.ptr<f32>
      %47 = llvm.select %40, %44, %46 : i1, f32
      %48 = llvm.getelementptr %arg20[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %47, %48 : !llvm.ptr<f32>
      llvm.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      llvm.return
    }
  }
}
