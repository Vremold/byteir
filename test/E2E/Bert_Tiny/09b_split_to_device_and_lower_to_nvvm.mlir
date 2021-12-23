// RUN: byteir-opt %s -collect-gpu-kernel -convert-scf-to-std -gpu-to-nvvm-ext -cse -reconcile-unrealized-casts -cse -cse -cse | FileCheck %s

// CHECK-LABEL: gpu.module @unified

module attributes {byre.container_module, gpu.container_module}  {
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<2x128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi64>, %arg4: memref<2x128xi64>, %arg5: memref<2x128xf64>, %arg6: memref<2x128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c256 = constant 256 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c256 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%16, %10] : memref<2x128xi64>
        %18 = trunci %17 : i64 to i32
        %19 = builtin.unrealized_conversion_cast %18 : i32 to ui32
        memref.store %19, %arg1[%16, %10] : memref<2x128xui32>
        %20 = memref.load %arg2[%16, %10] : memref<2x128xi64>
        %21 = memref.load %arg3[%16, %10] : memref<2x128xi64>
        %22 = addi %17, %21 : i64
        %23 = cmpi slt, %17, %20 : i64
        %24 = select %23, %22, %17 : i64
        memref.store %24, %arg4[%16, %10] : memref<2x128xi64>
        %25 = memref.load %arg5[%16, %10] : memref<2x128xf64>
        %26 = sitofp %17 : i64 to f64
        %27 = cmpf une, %26, %25 : f64
        memref.store %27, %arg6[%16, %10] : memref<2x128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown1_kernel {
    gpu.func @Unknown1_kernel(%arg0: memref<128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi64>, %arg4: memref<2x128xi64>, %arg5: memref<2x128xf64>, %arg6: memref<2x128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c256 = constant 256 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c256 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%10] : memref<128xi64>
        %18 = trunci %17 : i64 to i32
        %19 = builtin.unrealized_conversion_cast %18 : i32 to ui32
        memref.store %19, %arg1[%16, %10] : memref<2x128xui32>
        %20 = memref.load %arg2[%16, %10] : memref<2x128xi64>
        %21 = memref.load %arg3[%16, %10] : memref<2x128xi64>
        %22 = addi %17, %21 : i64
        %23 = cmpi slt, %17, %20 : i64
        %24 = select %23, %22, %17 : i64
        memref.store %24, %arg4[%16, %10] : memref<2x128xi64>
        %25 = memref.load %arg5[%16, %10] : memref<2x128xf64>
        %26 = sitofp %17 : i64 to f64
        %27 = cmpf une, %26, %25 : f64
        memref.store %27, %arg6[%16, %10] : memref<2x128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown2_kernel {
    gpu.func @Unknown2_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c32768 = constant 32768 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = remi_signed %16, %c128 : index
        %18 = cmpi slt, %17, %c0 : index
        %19 = addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = cmpi slt, %16, %c0 : index
        %22 = subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = divi_signed %23, %c128 : index
        %25 = subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = addf %27, %28 : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi64>, %arg4: memref<128xi64>, %arg5: memref<128xf64>, %arg6: memref<128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c128 = constant 128 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c128 : index
      scf.if %6 {
        %7 = memref.load %arg0[%5] : memref<128xi64>
        %8 = trunci %7 : i64 to i32
        %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
        memref.store %9, %arg1[%5] : memref<128xui32>
        %10 = memref.load %arg2[%5] : memref<128xi64>
        %11 = memref.load %arg3[%5] : memref<128xi64>
        %12 = addi %7, %11 : i64
        %13 = cmpi slt, %7, %10 : i64
        %14 = select %13, %12, %7 : i64
        memref.store %14, %arg4[%5] : memref<128xi64>
        %15 = memref.load %arg5[%5] : memref<128xf64>
        %16 = sitofp %7 : i64 to f64
        %17 = cmpf une, %16, %15 : f64
        memref.store %17, %arg6[%5] : memref<128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown4_kernel {
    gpu.func @Unknown4_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c32768 = constant 32768 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = remi_signed %16, %c128 : index
        %18 = cmpi slt, %17, %c0 : index
        %19 = addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = cmpi slt, %16, %c0 : index
        %22 = subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = divi_signed %23, %c128 : index
        %25 = subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = memref.load %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = memref.load %arg3[%26, %20, %10] : memref<2x128x128xf32>
        %31 = addf %27, %28 : f32
        %32 = addf %31, %29 : f32
        %33 = addf %32, %30 : f32
        memref.store %33, %arg4[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown5_kernel {
    gpu.func @Unknown5_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c32768 = constant 32768 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = remi_signed %16, %c128 : index
        %18 = cmpi slt, %17, %c0 : index
        %19 = addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = cmpi slt, %16, %c0 : index
        %22 = subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = divi_signed %23, %c128 : index
        %25 = subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = memref.load %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = memref.load %arg3[%26, %20, %10] : memref<2x128x128xf32>
        %31 = addf %27, %28 : f32
        %32 = addf %31, %29 : f32
        %33 = addf %32, %30 : f32
        memref.store %33, %arg4[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<2x128xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128xi1>, %arg5: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c32768 = constant 32768 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = remi_signed %16, %c128 : index
        %18 = cmpi slt, %17, %c0 : index
        %19 = addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = cmpi slt, %16, %c0 : index
        %22 = subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = divi_signed %23, %c128 : index
        %25 = subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20] : memref<2x128xi1>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = memref.load %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = select %27, %28, %29 : f32
        memref.store %30, %arg3[%26, %20, %10] : memref<2x128x128xf32>
        %31 = memref.load %arg4[%26, %20] : memref<2x128xi1>
        %32 = select %31, %28, %29 : f32
        memref.store %32, %arg5[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = constant 0 : index
      %c16384 = constant 16384 : index
      %3 = muli %0, %2 : index
      %4 = addi %3, %1 : index
      %5 = addi %c0, %4 : index
      %6 = cmpi slt, %4, %c16384 : index
      scf.if %6 {
        %c128 = constant 128 : index
        %7 = remi_signed %5, %c128 : index
        %8 = cmpi slt, %7, %c0 : index
        %9 = addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = constant -1 : index
        %11 = cmpi slt, %5, %c0 : index
        %12 = subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = divi_signed %13, %c128 : index
        %15 = subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%16] : memref<128xi1>
        %18 = memref.load %arg1[%16, %10] : memref<128x128xf32>
        %19 = memref.load %arg2[%16, %10] : memref<128x128xf32>
        %20 = select %17, %18, %19 : f32
        memref.store %20, %arg3[%16, %10] : memref<128x128xf32>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1x512xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<30522x128xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<2x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<512x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128x128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128x128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128x128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<2x1x1x128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<2x1x1x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128x128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512x128xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128x512xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128x128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<128xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<30522xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<2x128x30522xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg49: memref<30522x128xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg50: memref<2x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg51: memref<512x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg53: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg54: memref<128x128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg55: memref<128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg56: memref<128x128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg57: memref<128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg58: memref<128x128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg59: memref<128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg60: memref<128x128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg63: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg64: memref<512x128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg65: memref<512xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg66: memref<128x512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg69: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg70: memref<128x128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg71: memref<128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg72: memref<128x128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg73: memref<128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg74: memref<128x128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg75: memref<128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg76: memref<128x128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg79: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg80: memref<512x128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg81: memref<512xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg82: memref<128x512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg85: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg86: memref<128x128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg89: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg90: memref<30522xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<128x128xf32>
    %1 = memref.alloc() : memref<128x128xf32>
    %2 = memref.alloc() : memref<30522x128xf32>
    %3 = memref.alloc() : memref<2x128x128xf32>
    %4 = memref.alloc() : memref<2x128x128xf32>
    %5 = memref.alloc() : memref<2x128x128xf32>
    %6 = memref.alloc() : memref<2x128x128xf32>
    %7 = memref.alloc() : memref<2x128x128xf32>
    %8 = memref.alloc() : memref<2x2x128x64xf32>
    %9 = memref.alloc() : memref<2x2x128x64xf32>
    %10 = memref.alloc() : memref<2x2x128x128xf32>
    %11 = memref.alloc() : memref<2x2x128x64xf32>
    %12 = memref.alloc() : memref<2x2x128x128xf32>
    %13 = memref.alloc() : memref<2x2x128x64xf32>
    %14 = memref.alloc() : memref<2x128x2x64xf32>
    %15 = memref.alloc() : memref<2x128x128xf32>
    %16 = memref.alloc() : memref<2x128x128xf32>
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<2x128x128xf32>
    %19 = memref.alloc() : memref<2x128x128xf32>
    %20 = memref.alloc() : memref<2x128x512xf32>
    %21 = memref.alloc() : memref<2x128x128xf32>
    %22 = memref.alloc() : memref<2x128x128xf32>
    %23 = memref.alloc() : memref<2x128x128xf32>
    %24 = memref.alloc() : memref<2x128x128xf32>
    %25 = memref.alloc() : memref<2x128x128xf32>
    %26 = memref.alloc() : memref<2x2x128x64xf32>
    %27 = memref.alloc() : memref<2x2x128x64xf32>
    %28 = memref.alloc() : memref<2x2x128x128xf32>
    %29 = memref.alloc() : memref<2x2x128x64xf32>
    %30 = memref.alloc() : memref<2x2x128x128xf32>
    %31 = memref.alloc() : memref<2x2x128x64xf32>
    %32 = memref.alloc() : memref<2x128x2x64xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<2x128x128xf32>
    %36 = memref.alloc() : memref<2x128x128xf32>
    %37 = memref.alloc() : memref<2x128x128xf32>
    %38 = memref.alloc() : memref<2x128x512xf32>
    %39 = memref.alloc() : memref<2x128x128xf32>
    %40 = memref.alloc() : memref<2x128x128xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<2x128x128xf32>
    %43 = memref.alloc() : memref<30522x128xf32>
    %44 = memref.alloc() : memref<2x128x128xf32>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<256xf32>
    %47 = memref.alloc() : memref<2x128x128xf32>
    %48 = memref.alloc() : memref<0xf32>
    %49 = memref.alloc() : memref<2x128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<2x128x128xf32>
    %52 = memref.alloc() : memref<256xf32>
    %53 = memref.alloc() : memref<256xf32>
    %54 = memref.alloc() : memref<2x128x128xf32>
    %55 = memref.alloc() : memref<2x128x128xui8>
    %56 = memref.alloc() : memref<2x128x128xf32>
    %57 = memref.alloc() : memref<2x128x128xf32>
    %58 = memref.alloc() : memref<0xf32>
    %59 = memref.alloc() : memref<2x128x512xf32>
    %60 = memref.alloc() : memref<2x128x512xf32>
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<256xf32>
    %63 = memref.alloc() : memref<256xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    %65 = memref.alloc() : memref<2x128x128xui8>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<2x128x128xf32>
    %69 = memref.alloc() : memref<2x128x2x64xf32>
    %70 = memref.alloc() : memref<2x2x128x64xf32>
    %71 = memref.alloc() : memref<2x2x128x64xf32>
    %72 = memref.alloc() : memref<2x2x128x128xui8>
    %73 = memref.alloc() : memref<2x2x128x128xf32>
    %74 = memref.alloc() : memref<2x2x128x128xf32>
    %75 = memref.alloc() : memref<2x2x128x128xf32>
    %76 = memref.alloc() : memref<2x2x128x64xf32>
    %77 = memref.alloc() : memref<2x2x128x64xf32>
    %78 = memref.alloc() : memref<2x128x128xf32>
    %79 = memref.alloc() : memref<256xf32>
    %80 = memref.alloc() : memref<256xf32>
    %81 = memref.alloc() : memref<2x128x128xf32>
    %82 = memref.alloc() : memref<2x128x128xui8>
    %83 = memref.alloc() : memref<2x128x128xf32>
    %84 = memref.alloc() : memref<2x128x128xf32>
    %85 = memref.alloc() : memref<0xf32>
    %86 = memref.alloc() : memref<2x128x512xf32>
    %87 = memref.alloc() : memref<2x128x512xf32>
    %88 = memref.alloc() : memref<2x128x128xf32>
    %89 = memref.alloc() : memref<256xf32>
    %90 = memref.alloc() : memref<256xf32>
    %91 = memref.alloc() : memref<2x128x128xf32>
    %92 = memref.alloc() : memref<2x128x128xui8>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<2x128x128xf32>
    %95 = memref.alloc() : memref<2x128x128xf32>
    %96 = memref.alloc() : memref<2x128x2x64xf32>
    %97 = memref.alloc() : memref<2x2x128x64xf32>
    %98 = memref.alloc() : memref<2x2x128x64xf32>
    %99 = memref.alloc() : memref<2x2x128x128xui8>
    %100 = memref.alloc() : memref<2x2x128x128xf32>
    %101 = memref.alloc() : memref<2x2x128x128xf32>
    %102 = memref.alloc() : memref<2x2x128x128xf32>
    %103 = memref.alloc() : memref<2x2x128x64xf32>
    %104 = memref.alloc() : memref<2x2x128x64xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    %106 = memref.alloc() : memref<256xf32>
    %107 = memref.alloc() : memref<256xf32>
    %108 = memref.alloc() : memref<2x128x128xf32>
    %109 = memref.alloc() : memref<1x128x128xf32>
    %110 = memref.alloc() : memref<128x128xf32>
    %111 = memref.alloc() : memref<256x128xf32>
    %112 = memref.alloc() : memref<256x128xf32>
    %113 = memref.alloc() : memref<1x128xi64>
    %114 = memref.alloc() : memref<128xi64>
    %115 = memref.alloc() : memref<1x128xi64>
    %116 = memref.alloc() : memref<30522x128xf32>
    %117 = memref.alloc() : memref<256xi64>
    %118 = memref.alloc() : memref<256xf64>
    %119 = memref.alloc() : memref<2x128xf32>
    %120 = memref.alloc() : memref<256xi64>
    %121 = memref.alloc() : memref<256xi64>
    %122 = memref.alloc() : memref<256xf64>
    %123 = memref.alloc() : memref<256x128xf32>
    %124 = memref.alloc() : memref<512x128xf32>
    %125 = memref.alloc() : memref<128xi64>
    %126 = memref.alloc() : memref<128xi64>
    %127 = memref.alloc() : memref<128xf64>
    byre.compute @FillOp(%0) {value = dense<0.000000e+00> : tensor<128x128xf32>} : memref<128x128xf32>
    byre.compute @FillOp(%127) {value = dense<-1.000000e+00> : tensor<128xf64>} : memref<128xf64>
    byre.compute @FillOp(%126) {value = dense<512> : tensor<128xi64>} : memref<128xi64>
    byre.compute @FillOp(%125) {value = dense<0> : tensor<128xi64>} : memref<128xi64>
    byre.compute @FillOp(%124) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    byre.compute @FillOp(%123) {value = dense<0.000000e+00> : tensor<256x128xf32>} : memref<256x128xf32>
    byre.compute @FillOp(%122) {value = dense<-1.000000e+00> : tensor<256xf64>} : memref<256xf64>
    byre.compute @FillOp(%121) {value = dense<2> : tensor<256xi64>} : memref<256xi64>
    byre.compute @FillOp(%120) {value = dense<0> : tensor<256xi64>} : memref<256xi64>
    byre.compute @FillOp(%119) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    byre.compute @FillOp(%118) {value = dense<0.000000e+00> : tensor<256xf64>} : memref<256xf64>
    byre.compute @FillOp(%117) {value = dense<30522> : tensor<256xi64>} : memref<256xi64>
    byre.compute @FillOp(%116) {value = dense<0.000000e+00> : tensor<30522x128xf32>} : memref<30522x128xf32>
    byre.compute @AliasOp(%arg1, %115) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    byre.compute @AliasOp(%115, %114) {offset = 0 : i32} : memref<1x128xi64>, memref<128xi64>
    byre.compute @AliasOp(%arg2, %113) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %128 = memref.alloc() : memref<256xui32>
    %129 = memref.alloc() : memref<256x1xi64>
    %130 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg0, %120, %117, %118, %128, %129, %130) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_offsets = [0 : i32, 4 : i32, 1 : i32, 2 : i32, 5 : i32, 3 : i32, 6 : i32], arg_ranks = [2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown0_kernel"} : memref<2x128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg3, %128, %112) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %131 = memref.alloc() : memref<256xui32>
    %132 = memref.alloc() : memref<256x1xi64>
    %133 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%114, %120, %121, %122, %131, %132, %133) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_offsets = [0 : i32, 4 : i32, 1 : i32, 2 : i32, 5 : i32, 3 : i32, 6 : i32], arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown1_kernel"} : memref<128xi64>, memref<256xi64>, memref<256xi64>, memref<256xf64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg4, %131, %111) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %134 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%112, %111, %134) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown2_kernel"} : memref<256x128xf32>, memref<256x128xf32>, memref<2x128x128xf32>
    %135 = memref.alloc() : memref<128xui32>
    %136 = memref.alloc() : memref<128x1xi64>
    %137 = memref.alloc() : memref<128xi1>
    byre.compute @PTXOp(%113, %125, %126, %127, %135, %136, %137) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_offsets = [0 : i32, 4 : i32, 1 : i32, 2 : i32, 5 : i32, 3 : i32, 6 : i32], arg_ranks = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3_kernel"} : memref<1x128xi64>, memref<128xi64>, memref<128xi64>, memref<128xf64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    byre.compute @IndexSelectOp(%arg5, %135, %110) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    byre.compute @AliasOp(%110, %109) {offset = 0 : i32} : memref<128x128xf32>, memref<1x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%134, %arg6, %arg7, %109, %108, %107, %106, %105) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose(%108, %arg8, %arg9, %104) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%108, %arg10, %arg11, %103) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%104, %103, %102) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%102, %arg14, %101, %100, %99) {batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32} : memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%108, %arg12, %arg13, %98) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%100, %98, %97) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%97, %96) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%96, %95) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%95, %arg15, %arg16, %94, %93, %92) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%94, %arg17, %arg18, %108, %91, %90, %89, %88) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%91, %arg19, %arg20, %87, %86, %85) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%87, %arg21, %arg22, %84, %83, %82) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%84, %arg23, %arg24, %91, %81, %80, %79, %78) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose(%81, %arg25, %arg26, %77) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%81, %arg27, %arg28, %76) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%77, %76, %75) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%75, %arg31, %74, %73, %72) {batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 128 : i32} : memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%81, %arg29, %arg30, %71) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%73, %71, %70) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%70, %69) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%69, %68) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%68, %arg32, %arg33, %67, %66, %65) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%67, %arg34, %arg35, %81, %64, %63, %62, %61) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%64, %arg36, %arg37, %60, %59, %58) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%60, %arg38, %arg39, %57, %56, %55) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%57, %arg40, %arg41, %64, %54, %53, %52, %51) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%54, %arg42, %arg43, %50, %49, %48) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    byre.compute @ftv4.layernorm(%50, %arg44, %arg45, %47, %46, %45) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @ftv4.linear(%47, %arg3, %arg46, %arg48) : memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>, memref<2x128x30522xf32>
    byre.compute @ftv4.linear_backward(%arg47, %47, %arg3, %44, %43, %arg90) : memref<2x128x30522xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<2x128x128xf32>, memref<30522x128xf32>, memref<30522xf32>
    byre.compute @ftv4.layernorm_backward(%44, %50, %arg44, %46, %45, %42, %arg88, %arg89) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%42, %54, %arg42, %49, %48, %41, %arg86, %arg87) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%41, %51, %arg40, %53, %52, %40, %arg84, %arg85, %39) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%40, %60, %arg38, %56, %55, %38, %arg82, %arg83) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%38, %64, %arg36, %59, %58, %37, %arg80, %arg81) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @AddOp(%39, %37, %36) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%36, %61, %arg34, %63, %62, %35, %arg78, %arg79, %34) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%35, %68, %arg32, %66, %65, %33, %arg76, %arg77) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%33, %32) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%32, %31) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%31, %73, %71, %30, %29) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%30, %74, %72, %28) {dropout_rate = 1.000000e-01 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%28, %77, %76, %27, %26) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%27, %81, %arg25, %25, %arg70, %arg71) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%29, %81, %arg29, %24, %arg74, %arg75) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%26, %81, %arg27, %23, %arg72, %arg73) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %138 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%34, %25, %24, %23, %138) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown4_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%138, %78, %arg23, %80, %79, %22, %arg68, %arg69, %21) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%22, %87, %arg21, %83, %82, %20, %arg66, %arg67) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%20, %91, %arg19, %86, %85, %19, %arg64, %arg65) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @AddOp(%21, %19, %18) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%18, %88, %arg17, %90, %89, %17, %arg62, %arg63, %16) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%17, %95, %arg15, %93, %92, %15, %arg60, %arg61) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%15, %14) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%14, %13) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%13, %100, %98, %12, %11) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%12, %101, %99, %10) {dropout_rate = 1.000000e-01 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%10, %104, %103, %9, %8) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%9, %108, %arg8, %7, %arg54, %arg55) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%11, %108, %arg12, %6, %arg58, %arg59) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%8, %108, %arg10, %5, %arg56, %arg57) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %139 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%16, %7, %6, %5, %139) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown5_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%139, %105, %arg6, %107, %106, %4, %arg52, %arg53, %3) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %140 = memref.alloc() : memref<256x128xf32>
    %141 = memref.alloc() : memref<256x128xf32>
    byre.compute @PTXOp(%130, %4, %123, %133, %140, %141) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 4 : i32, 3 : i32, 5 : i32], arg_ranks = [2 : i32, 3 : i32, 3 : i32, 3 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown6_kernel"} : memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOp(%116, %129, %140, %2) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @AddOp(%43, %2, %arg49) : memref<30522x128xf32>, memref<30522x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOp(%119, %132, %141, %arg50) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    byre.compute @ReduceSumOp(%3, %1) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %142 = memref.alloc() : memref<128x128xf32>
    byre.compute @PTXOp(%137, %1, %0, %142) {BlockSize.x = 32 : i32, GridSize.x = 512 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown7_kernel"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOp(%124, %136, %142, %arg51) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    return
  }
}

