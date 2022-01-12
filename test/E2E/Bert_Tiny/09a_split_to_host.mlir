// RUN: byteir-opt %s -collect-func="anchor-attr=byre.entry_point" -func-tag="attach-attr=device_file_name:String:your_file func-name=main" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module}  {
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<2x128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c30522_i64 = arith.constant 30522 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant 0.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c256 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%16, %10] : memref<2x128xi64>
        %18 = arith.trunci %17 : i64 to i32
        %19 = builtin.unrealized_conversion_cast %18 : i32 to ui32
        memref.store %19, %arg1[%16, %10] : memref<2x128xui32>
        %20 = arith.addi %17, %c30522_i64 : i64
        %21 = arith.cmpi slt, %17, %c0_i64 : i64
        %22 = select %21, %20, %17 : i64
        memref.store %22, %arg2[%16, %10] : memref<2x128xi64>
        %23 = arith.sitofp %17 : i64 to f64
        %24 = arith.cmpf une, %23, %cst : f64
        memref.store %24, %arg3[%16, %10] : memref<2x128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown1_kernel {
    gpu.func @Unknown1_kernel(%arg0: memref<128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c2_i64 = arith.constant 2 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant -1.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c256 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%10] : memref<128xi64>
        %18 = arith.trunci %17 : i64 to i32
        %19 = builtin.unrealized_conversion_cast %18 : i32 to ui32
        memref.store %19, %arg1[%16, %10] : memref<2x128xui32>
        %20 = arith.addi %17, %c2_i64 : i64
        %21 = arith.cmpi slt, %17, %c0_i64 : i64
        %22 = select %21, %20, %17 : i64
        memref.store %22, %arg2[%16, %10] : memref<2x128xi64>
        %23 = arith.sitofp %17 : i64 to f64
        %24 = arith.cmpf une, %23, %cst : f64
        memref.store %24, %arg3[%16, %10] : memref<2x128xi1>
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
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = arith.addf %27, %28 : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi1>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c512_i64 = arith.constant 512 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant -1.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c128 : index
      scf.if %6 {
        %7 = memref.load %arg0[%5] : memref<128xi64>
        %8 = arith.trunci %7 : i64 to i32
        %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
        memref.store %9, %arg1[%5] : memref<128xui32>
        %10 = arith.addi %7, %c512_i64 : i64
        %11 = arith.cmpi slt, %7, %c0_i64 : i64
        %12 = select %11, %10, %7 : i64
        memref.store %12, %arg2[%5] : memref<128xi64>
        %13 = arith.sitofp %7 : i64 to f64
        %14 = arith.cmpf une, %13, %cst : f64
        memref.store %14, %arg3[%5] : memref<128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown4_kernel {
    gpu.func @Unknown4_kernel(%arg0: memref<2x128x30522xf32>, %arg1: memref<30522xf32>, %arg2: memref<2x128x30522xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %6 {
        %c30522 = arith.constant 30522 : index
        %7 = arith.remsi %5, %c30522 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c30522 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c30522 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %c128 = arith.constant 128 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x30522xf32>
        %28 = memref.load %arg1[%10] : memref<30522xf32>
        %29 = arith.addf %27, %28 : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x30522xf32>
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
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = memref.load %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = memref.load %arg3[%26, %20, %10] : memref<2x128x128xf32>
        %31 = arith.addf %27, %28 : f32
        %32 = arith.addf %31, %29 : f32
        %33 = arith.addf %32, %30 : f32
        memref.store %33, %arg4[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20, %10] : memref<2x128x128xf32>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = memref.load %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = memref.load %arg3[%26, %20, %10] : memref<2x128x128xf32>
        %31 = arith.addf %27, %28 : f32
        %32 = arith.addf %31, %29 : f32
        %33 = arith.addf %32, %30 : f32
        memref.store %33, %arg4[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<2x128xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128xi1>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = select %21, %25, %24 : index
        %27 = memref.load %arg0[%26, %20] : memref<2x128xi1>
        %28 = memref.load %arg1[%26, %20, %10] : memref<2x128x128xf32>
        %29 = select %27, %28, %cst : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x128xf32>
        %30 = memref.load %arg3[%26, %20] : memref<2x128xi1>
        %31 = select %30, %28, %cst : f32
        memref.store %31, %arg4[%26, %20, %10] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown8_kernel {
    gpu.func @Unknown8_kernel(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %2 = "gpu.block_dim"() {dimension = "x"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %cst = arith.constant 0.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.addi %c0, %4 : index
      %6 = arith.cmpi slt, %4, %c16384 : index
      scf.if %6 {
        %c128 = arith.constant 128 : index
        %7 = arith.remsi %5, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = select %8, %9, %7 : index
        %c-1 = arith.constant -1 : index
        %11 = arith.cmpi slt, %5, %c0 : index
        %12 = arith.subi %c-1, %5 : index
        %13 = select %11, %12, %5 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = select %11, %15, %14 : index
        %17 = memref.load %arg0[%16] : memref<128xi1>
        %18 = memref.load %arg1[%16, %10] : memref<128x128xf32>
        %19 = select %17, %18, %cst : f32
        memref.store %19, %arg2[%16, %10] : memref<128x128xf32>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1x512xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<30522x128xf32> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<2x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<512x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128x128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128x128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128x128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<2x1x1x128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<2x1x1x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128x128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512x128xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128x512xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128x128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<128xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<30522xf32> {byre.argname = "Input46", byre.argtype = 1 : i32}, %arg47: memref<2x128x30522xf32> {byre.argname = "Input47", byre.argtype = 1 : i32}, %arg48: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg49: memref<30522x128xf32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg50: memref<2x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg51: memref<512x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg53: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg54: memref<128x128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg55: memref<128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg56: memref<128x128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg57: memref<128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg58: memref<128x128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg59: memref<128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg60: memref<128x128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg63: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg64: memref<512x128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg65: memref<512xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg66: memref<128x512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg69: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg70: memref<128x128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg71: memref<128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg72: memref<128x128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg73: memref<128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg74: memref<128x128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg75: memref<128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg76: memref<128x128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg79: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg80: memref<512x128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg81: memref<512xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg82: memref<128x512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg85: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg86: memref<128x128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg89: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg90: memref<30522xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<2x128xf32>
    %1 = memref.alloc() : memref<128x128xf32>
    %2 = memref.alloc() : memref<2x128x128xf32>
    %3 = memref.alloc() : memref<2x128x128xf32>
    %4 = memref.alloc() : memref<2x128x128xf32>
    %5 = memref.alloc() : memref<2x128x128xf32>
    %6 = memref.alloc() : memref<2x128x128xf32>
    %7 = memref.alloc() : memref<2x2x128x64xf32>
    %8 = memref.alloc() : memref<2x2x128x64xf32>
    %9 = memref.alloc() : memref<2x2x128x128xf32>
    %10 = memref.alloc() : memref<2x2x128x64xf32>
    %11 = memref.alloc() : memref<2x2x128x128xf32>
    %12 = memref.alloc() : memref<2x2x128x64xf32>
    %13 = memref.alloc() : memref<2x128x2x64xf32>
    %14 = memref.alloc() : memref<2x128x128xf32>
    %15 = memref.alloc() : memref<2x128x128xf32>
    %16 = memref.alloc() : memref<2x128x128xf32>
    %17 = memref.alloc() : memref<2x128x128xf32>
    %18 = memref.alloc() : memref<2x128x128xf32>
    %19 = memref.alloc() : memref<2x128x512xf32>
    %20 = memref.alloc() : memref<2x128x128xf32>
    %21 = memref.alloc() : memref<2x128x128xf32>
    %22 = memref.alloc() : memref<2x128x128xf32>
    %23 = memref.alloc() : memref<2x128x128xf32>
    %24 = memref.alloc() : memref<2x128x128xf32>
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    %26 = memref.alloc() : memref<2x2x128x64xf32>
    %27 = memref.alloc() : memref<2x2x128x128xf32>
    %28 = memref.alloc() : memref<2x2x128x64xf32>
    %29 = memref.alloc() : memref<2x2x128x128xf32>
    %30 = memref.alloc() : memref<2x2x128x64xf32>
    %31 = memref.alloc() : memref<2x128x2x64xf32>
    %32 = memref.alloc() : memref<2x128x128xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    %34 = memref.alloc() : memref<2x128x128xf32>
    %35 = memref.alloc() : memref<2x128x128xf32>
    %36 = memref.alloc() : memref<2x128x128xf32>
    %37 = memref.alloc() : memref<2x128x512xf32>
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<2x128x128xf32>
    %40 = memref.alloc() : memref<2x128x128xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    %42 = memref.alloc() : memref<256x30522xf32>
    %43 = memref.alloc() : memref<256x128xf32>
    %44 = memref.alloc() : memref<256xf32>
    %45 = memref.alloc() : memref<256xf32>
    %46 = memref.alloc() : memref<2x128x128xf32>
    %47 = memref.alloc() : memref<0xf32>
    %48 = memref.alloc() : memref<2x128x128xf32>
    %49 = memref.alloc() : memref<2x128x128xf32>
    %50 = memref.alloc() : memref<2x128x128xf32>
    %51 = memref.alloc() : memref<256xf32>
    %52 = memref.alloc() : memref<256xf32>
    %53 = memref.alloc() : memref<2x128x128xf32>
    %54 = memref.alloc() : memref<2x128x128xui8>
    %55 = memref.alloc() : memref<2x128x128xf32>
    %56 = memref.alloc() : memref<2x128x128xf32>
    %57 = memref.alloc() : memref<0xf32>
    %58 = memref.alloc() : memref<2x128x512xf32>
    %59 = memref.alloc() : memref<2x128x512xf32>
    %60 = memref.alloc() : memref<2x128x128xf32>
    %61 = memref.alloc() : memref<256xf32>
    %62 = memref.alloc() : memref<256xf32>
    %63 = memref.alloc() : memref<2x128x128xf32>
    %64 = memref.alloc() : memref<2x128x128xui8>
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
    %81 = memref.alloc() : memref<2x128x128xui8>
    %82 = memref.alloc() : memref<2x128x128xf32>
    %83 = memref.alloc() : memref<2x128x128xf32>
    %84 = memref.alloc() : memref<0xf32>
    %85 = memref.alloc() : memref<2x128x512xf32>
    %86 = memref.alloc() : memref<2x128x512xf32>
    %87 = memref.alloc() : memref<2x128x128xf32>
    %88 = memref.alloc() : memref<256xf32>
    %89 = memref.alloc() : memref<256xf32>
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<2x128x128xui8>
    %92 = memref.alloc() : memref<2x128x128xf32>
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<2x128x128xf32>
    %95 = memref.alloc() : memref<2x128x2x64xf32>
    %96 = memref.alloc() : memref<2x2x128x64xf32>
    %97 = memref.alloc() : memref<2x2x128x64xf32>
    %98 = memref.alloc() : memref<2x2x128x128xui8>
    %99 = memref.alloc() : memref<2x2x128x128xf32>
    %100 = memref.alloc() : memref<2x2x128x128xf32>
    %101 = memref.alloc() : memref<2x2x128x128xf32>
    %102 = memref.alloc() : memref<2x2x128x64xf32>
    %103 = memref.alloc() : memref<2x2x128x64xf32>
    %104 = memref.alloc() : memref<2x128x128xf32>
    %105 = memref.alloc() : memref<256xf32>
    %106 = memref.alloc() : memref<256xf32>
    %107 = memref.alloc() : memref<2x128x128xf32>
    %108 = memref.alloc() : memref<1x128x128xf32>
    %109 = memref.alloc() : memref<128x128xf32>
    %110 = memref.alloc() : memref<256x128xf32>
    %111 = memref.alloc() : memref<2x128x128xf32>
    %112 = memref.alloc() : memref<256x128xf32>
    %113 = memref.alloc() : memref<256x128xf32>
    %114 = memref.alloc() : memref<256x30522xf32>
    %115 = memref.alloc() : memref<1x128xi64>
    %116 = memref.alloc() : memref<128xi64>
    %117 = memref.alloc() : memref<1x128xi64>
    %118 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%0) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    byre.compute @FillOp(%118) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    byre.compute @AliasOp(%arg1, %117) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    byre.compute @AliasOp(%arg1, %116) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<128xi64>
    byre.compute @AliasOp(%arg2, %115) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    byre.compute @AliasOp(%arg47, %114) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256x30522xf32>
    %119 = memref.alloc() : memref<256xui32>
    %120 = memref.alloc() : memref<256x1xi64>
    %121 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg0, %119, %120, %121) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], arg_ranks = [2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown0_kernel"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg3, %119, %113) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    byre.compute @MatmulOp(%114, %arg3, %112) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    byre.compute @AliasOp(%112, %111) {offset = 0 : i32} : memref<256x128xf32>, memref<2x128x128xf32>
    %122 = memref.alloc() : memref<256xui32>
    %123 = memref.alloc() : memref<256x1xi64>
    %124 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%116, %122, %123, %124) {BlockSize.x = 32 : i32, GridSize.x = 8 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown1_kernel"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg4, %122, %110) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %125 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%113, %110, %125) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown2_kernel"} : memref<256x128xf32>, memref<256x128xf32>, memref<2x128x128xf32>
    %126 = memref.alloc() : memref<128xui32>
    %127 = memref.alloc() : memref<128x1xi64>
    %128 = memref.alloc() : memref<128xi1>
    byre.compute @PTXOp(%115, %126, %127, %128) {BlockSize.x = 32 : i32, GridSize.x = 4 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], arg_ranks = [1 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3_kernel"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    byre.compute @IndexSelectOp(%arg5, %126, %109) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    byre.compute @AliasOp(%109, %108) {offset = 0 : i32} : memref<128x128xf32>, memref<1x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%125, %arg6, %arg7, %108, %107, %106, %105, %104) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose(%107, %arg8, %arg9, %103) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%107, %arg10, %arg11, %102) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%103, %102, %101) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%101, %arg14, %100, %99, %98) {batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%107, %arg12, %arg13, %97) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%99, %97, %96) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%96, %95) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%95, %94) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%94, %arg15, %arg16, %93, %92, %91) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%93, %arg17, %arg18, %107, %90, %89, %88, %87) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%90, %arg19, %arg20, %86, %85, %84) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%86, %arg21, %arg22, %83, %82, %81) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%83, %arg23, %arg24, %90, %80, %79, %78, %77) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose(%80, %arg25, %arg26, %76) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%80, %arg27, %arg28, %75) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%76, %75, %74) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax(%74, %arg31, %73, %72, %71) {batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    byre.compute @ftv4.linear_transpose(%80, %arg29, %arg30, %70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%72, %70, %69) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d(%69, %68) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    byre.compute @AliasOp(%68, %67) {offset = 0 : i32} : memref<2x128x2x64xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%67, %arg32, %arg33, %66, %65, %64) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%66, %arg34, %arg35, %80, %63, %62, %61, %60) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%63, %arg36, %arg37, %59, %58, %57) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%59, %arg38, %arg39, %56, %55, %54) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>
    byre.compute @ftv4.layernorm_residual(%56, %arg40, %arg41, %63, %53, %52, %51, %50) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout(%53, %arg42, %arg43, %49, %48, %47) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    byre.compute @ftv4.layernorm(%49, %arg44, %arg45, %46, %45, %44) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    byre.compute @AliasOp(%46, %43) {offset = 0 : i32} : memref<2x128x128xf32>, memref<256x128xf32>
    byre.compute @MatmulOp(%43, %arg3, %42) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    byre.compute @PTXOp(%42, %arg46, %arg48) {BlockSize.x = 32 : i32, GridSize.x = 244176 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32], arg_ranks = [3 : i32, 1 : i32, 3 : i32], kernel_name = "Unknown4_kernel"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>
    %129 = memref.alloc() : memref<30522x128xf32>
    byre.compute @MatmulOp(%43, %114, %129) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    byre.compute @ftv4.layernorm_backward(%111, %49, %arg44, %45, %44, %41, %arg88, %arg89) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%41, %53, %arg42, %48, %47, %40, %arg86, %arg87) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%40, %50, %arg40, %52, %51, %39, %arg84, %arg85, %38) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%39, %59, %arg38, %55, %54, %37, %arg82, %arg83) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%37, %63, %arg36, %58, %57, %36, %arg80, %arg81) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @AddOp(%38, %36, %35) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%35, %60, %arg34, %62, %61, %34, %arg78, %arg79, %33) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%34, %67, %arg32, %65, %64, %32, %arg76, %arg77) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%32, %31) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%31, %30) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%30, %72, %70, %29, %28) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%29, %73, %71, %27) {dropout_rate = 1.000000e-01 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%27, %76, %75, %26, %25) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%26, %80, %arg25, %24, %arg70, %arg71) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%28, %80, %arg29, %23, %arg74, %arg75) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%25, %80, %arg27, %22, %arg72, %arg73) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %130 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%33, %24, %23, %22, %130) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown5_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%130, %77, %arg23, %79, %78, %21, %arg68, %arg69, %20) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%21, %86, %arg21, %82, %81, %19, %arg66, %arg67) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%19, %90, %arg19, %85, %84, %18, %arg64, %arg65) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @AddOp(%20, %18, %17) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%17, %87, %arg17, %89, %88, %16, %arg62, %arg63, %15) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%16, %94, %arg15, %92, %91, %14, %arg60, %arg61) {act_gelu = false, dropout_rate = 1.000000e-01 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @AliasOp(%14, %13) {offset = 0 : i32} : memref<2x128x128xf32>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d_backward(%13, %12) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%12, %99, %97, %11, %10) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%11, %100, %98, %9) {dropout_rate = 1.000000e-01 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul_backward(%9, %103, %102, %8, %7) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%8, %107, %arg8, %6, %arg54, %arg55) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%10, %107, %arg12, %5, %arg58, %arg59) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%7, %107, %arg10, %4, %arg56, %arg57) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%15, %6, %5, %4, %131) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown6_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%131, %104, %arg6, %106, %105, %3, %arg52, %arg53, %2) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %132 = memref.alloc() : memref<256x128xf32>
    %133 = memref.alloc() : memref<256x128xf32>
    byre.compute @PTXOp(%121, %3, %124, %132, %133) {BlockSize.x = 32 : i32, GridSize.x = 1024 : i32, arg_offsets = [0 : i32, 1 : i32, 3 : i32, 2 : i32, 4 : i32], arg_ranks = [2 : i32, 3 : i32, 3 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown7_kernel"} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOp(%129, %120, %132, %arg49) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOp(%0, %123, %133, %arg50) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    byre.compute @ReduceSumOp(%2, %1) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %134 = memref.alloc() : memref<128x128xf32>
    byre.compute @PTXOp(%128, %1, %134) {BlockSize.x = 32 : i32, GridSize.x = 512 : i32, arg_offsets = [0 : i32, 1 : i32, 2 : i32], arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8_kernel"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOp(%118, %127, %134, %arg51) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOp(%arg47, %arg90) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

