// RUN: byteir-opt %s -byre-host="device-file-name=your_file" | FileCheck %s

// CHECK-LABEL: func @main
module attributes {byre.container_module, gpu.container_module} {
  gpu.module @Unknown0_kernel {
    gpu.func @Unknown0_kernel(%arg0: memref<2x128xi64>, %arg1: memref<256xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %c-100_i64 = arith.constant -100 : i64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.cmpi ne, %16, %c-100_i64 : i64
        memref.store %17, %arg1[%4] : memref<256xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown1_kernel {
    gpu.func @Unknown1_kernel(%arg0: memref<2x128xi64>, %arg1: memref<256xui32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %c30522_i64 = arith.constant 30522 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant 0.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.trunci %16 : i64 to i32
        %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
        memref.store %18, %arg1[%4] : memref<256xui32>
        %19 = arith.addi %16, %c30522_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = select %20, %19, %16 : i64
        memref.store %21, %arg2[%4] : memref<256xi64>
        %22 = arith.sitofp %16 : i64 to f64
        %23 = arith.cmpf une, %22, %cst : f64
        memref.store %23, %arg3[%4] : memref<256xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown2_kernel {
    gpu.func @Unknown2_kernel(%arg0: memref<128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %c2_i64 = arith.constant 2 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant -1.000000e+00 : f64
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%9] : memref<128xi64>
        %17 = arith.trunci %16 : i64 to i32
        %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
        memref.store %18, %arg1[%15, %9] : memref<2x128xui32>
        %19 = arith.addi %16, %c2_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = select %20, %19, %16 : i64
        memref.store %21, %arg2[%15, %9] : memref<2x128xi64>
        %22 = arith.sitofp %16 : i64 to f64
        %23 = arith.cmpf une, %22, %cst : f64
        memref.store %23, %arg3[%15, %9] : memref<2x128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown3_kernel {
    gpu.func @Unknown3_kernel(%arg0: memref<128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi1>) kernel {
      %0 = gpu.thread_id  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c128 = arith.constant 128 : index
      %c512_i64 = arith.constant 512 : i64
      %c0_i64 = arith.constant 0 : i64
      %cst = arith.constant -1.000000e+00 : f64
      %1 = arith.cmpi slt, %0, %c128 : index
      scf.if %1 {
        %2 = memref.load %arg0[%0] : memref<128xi64>
        %3 = arith.trunci %2 : i64 to i32
        %4 = builtin.unrealized_conversion_cast %3 : i32 to ui32
        memref.store %4, %arg1[%0] : memref<128xui32>
        %5 = arith.addi %2, %c512_i64 : i64
        %6 = arith.cmpi slt, %2, %c0_i64 : i64
        %7 = select %6, %5, %2 : i64
        memref.store %7, %arg2[%0] : memref<128xi64>
        %8 = arith.sitofp %2 : i64 to f64
        %9 = arith.cmpf une, %8, %cst : f64
        memref.store %9, %arg3[%0] : memref<128xi1>
      }
      gpu.return
    }
  }
  gpu.module @Unknown4_kernel {
    gpu.func @Unknown4_kernel(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<2x128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%15, %9] : memref<256x128xf32>
        %27 = memref.load %arg1[%15, %9] : memref<256x128xf32>
        %28 = memref.load %arg2[%19, %9] : memref<128x128xf32>
        %29 = arith.addf %26, %27 : f32
        %30 = arith.addf %29, %28 : f32
        memref.store %30, %arg3[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown5_kernel {
    gpu.func @Unknown5_kernel(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>, %arg2: memref<2x128x30522xf32>, %arg3: memref<2x128x30522xf32>, %arg4: memref<256x30522xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c7813632 = arith.constant 7813632 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %c30522 = arith.constant 30522 : index
        %6 = arith.remsi %4, %c30522 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %c128 = arith.constant 128 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%15, %9] : memref<256x30522xf32>
        %27 = memref.load %arg1[%9] : memref<30522xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x30522xf32>
        %29 = memref.load %arg3[%25, %19, %9] : memref<2x128x30522xf32>
        %30 = arith.addf %29, %27 : f32
        memref.store %30, %arg4[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown6_kernel {
    gpu.func @Unknown6_kernel(%arg0: memref<256x30522xf32>, %arg1: memref<256xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c7813632 = arith.constant 7813632 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %c30522 = arith.constant 30522 : index
        %6 = arith.remsi %4, %c30522 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg1[%15] : memref<256xf32>
        %18 = arith.subf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
        %19 = math.exp %18 : f32
        memref.store %19, %arg3[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown7_kernel {
    gpu.func @Unknown7_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = arith.constant 256 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = math.log %6 : f32
        memref.store %7, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown8_kernel {
    gpu.func @Unknown8_kernel(%arg0: memref<256xi1>, %arg1: memref<256xi64>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>, %arg4: memref<256xf32>, %arg5: memref<256x30522xf32>, %arg6: memref<256x30522xf32>, %arg7: memref<256x30522xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c7813632 = arith.constant 7813632 : index
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %c30522 = arith.constant 30522 : index
        %6 = arith.remsi %4, %c30522 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15] : memref<256xi1>
        %17 = memref.load %arg1[%15] : memref<256xi64>
        %18 = arith.index_cast %9 : index to i64
        %19 = arith.cmpi eq, %17, %18 : i64
        %20 = select %19, %cst, %cst_0 : f32
        %21 = select %16, %cst, %cst_0 : f32
        %22 = arith.mulf %21, %20 : f32
        memref.store %22, %arg2[%15, %9] : memref<256x30522xf32>
        %23 = memref.load %arg3[%15, %9] : memref<256x30522xf32>
        %24 = memref.load %arg4[%15] : memref<256xf32>
        %25 = arith.subf %23, %24 : f32
        %26 = arith.negf %20 : f32
        %27 = arith.mulf %26, %25 : f32
        %28 = arith.cmpf une, %20, %cst : f32
        %29 = select %28, %cst_0, %27 : f32
        %30 = arith.mulf %29, %22 : f32
        memref.store %30, %arg5[%15, %9] : memref<256x30522xf32>
        %31 = arith.mulf %26, %22 : f32
        memref.store %31, %arg6[%15, %9] : memref<256x30522xf32>
        %32 = math.exp %25 : f32
        memref.store %32, %arg7[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown9_kernel {
    gpu.func @Unknown9_kernel(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
      %0 = gpu.thread_id  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 1.000000e+00 : f32
      %1 = arith.cmpi slt, %0, %c1 : index
      scf.if %1 {
        %2 = memref.load %arg0[] : memref<f32>
        %3 = arith.cmpf une, %2, %cst : f32
        %4 = select %3, %2, %cst_0 : f32
        memref.store %4, %arg1[] : memref<f32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown10_kernel {
    gpu.func @Unknown10_kernel(%arg0: memref<256x30522xf32>, %arg1: memref<f32>, %arg2: memref<256x30522xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c7813632 = arith.constant 7813632 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %c30522 = arith.constant 30522 : index
        %6 = arith.remsi %4, %c30522 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg1[] : memref<f32>
        %18 = arith.divf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown11_kernel {
    gpu.func @Unknown11_kernel(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) kernel {
      %0 = gpu.thread_id  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %1 = arith.cmpi slt, %0, %c1 : index
      scf.if %1 {
        %2 = memref.load %arg0[] : memref<f32>
        %3 = memref.load %arg1[] : memref<f32>
        %4 = arith.divf %2, %3 : f32
        memref.store %4, %arg2[] : memref<f32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown12_kernel {
    gpu.func @Unknown12_kernel(%arg0: memref<256x30522xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xf32>, %arg3: memref<256x30522xf32>, %arg4: memref<2x128xf32>, %arg5: memref<2x128x30522xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c7813632 = arith.constant 7813632 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %c30522 = arith.constant 30522 : index
        %6 = arith.remsi %4, %c30522 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %18 = memref.load %arg2[%15] : memref<256xf32>
        %19 = arith.mulf %17, %18 : f32
        %20 = arith.subf %16, %19 : f32
        memref.store %20, %arg3[%15, %9] : memref<256x30522xf32>
        %c128 = arith.constant 128 : index
        %21 = arith.remsi %15, %c128 : index
        %22 = arith.cmpi slt, %21, %c0 : index
        %23 = arith.addi %21, %c128 : index
        %24 = select %22, %23, %21 : index
        %25 = arith.cmpi slt, %15, %c0 : index
        %26 = arith.subi %c-1, %15 : index
        %27 = select %25, %26, %15 : index
        %28 = arith.divsi %27, %c128 : index
        %29 = arith.subi %c-1, %28 : index
        %30 = select %25, %29, %28 : index
        %31 = memref.load %arg4[%30, %24] : memref<2x128xf32>
        %32 = arith.mulf %17, %31 : f32
        %33 = arith.subf %16, %32 : f32
        memref.store %33, %arg5[%30, %24, %9] : memref<2x128x30522xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown13_kernel {
    gpu.func @Unknown13_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown14_kernel {
    gpu.func @Unknown14_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = memref.load %arg2[%25, %19, %9] : memref<2x128x128xf32>
        %29 = memref.load %arg3[%25, %19, %9] : memref<2x128x128xf32>
        %30 = arith.addf %26, %27 : f32
        %31 = arith.addf %30, %28 : f32
        %32 = arith.addf %31, %29 : f32
        memref.store %32, %arg4[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown15_kernel {
    gpu.func @Unknown15_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown16_kernel {
    gpu.func @Unknown16_kernel(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = memref.load %arg2[%25, %19, %9] : memref<2x128x128xf32>
        %29 = memref.load %arg3[%25, %19, %9] : memref<2x128x128xf32>
        %30 = arith.addf %26, %27 : f32
        %31 = arith.addf %30, %28 : f32
        %32 = arith.addf %31, %29 : f32
        memref.store %32, %arg4[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown17_kernel {
    gpu.func @Unknown17_kernel(%arg0: memref<2x128xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256x128xf32>, %arg3: memref<2x128xi1>, %arg4: memref<256x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19] : memref<2x128xi1>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = select %26, %27, %cst : f32
        memref.store %28, %arg2[%15, %9] : memref<256x128xf32>
        %29 = memref.load %arg3[%25, %19] : memref<2x128xi1>
        %30 = select %29, %27, %cst : f32
        memref.store %30, %arg4[%15, %9] : memref<256x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @Unknown18_kernel {
    gpu.func @Unknown18_kernel(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.block_dim  x
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c16384 = arith.constant 16384 : index
      %cst = arith.constant 0.000000e+00 : f32
      %3 = arith.muli %0, %2 : index
      %4 = arith.addi %3, %1 : index
      %5 = arith.cmpi slt, %4, %c16384 : index
      scf.if %5 {
        %c128 = arith.constant 128 : index
        %6 = arith.remsi %4, %c128 : index
        %c0 = arith.constant 0 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = select %7, %8, %6 : index
        %c-1 = arith.constant -1 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = select %10, %14, %13 : index
        %16 = memref.load %arg0[%15] : memref<128xi1>
        %17 = memref.load %arg1[%15, %9] : memref<128x128xf32>
        %18 = select %16, %17, %cst : f32
        memref.store %18, %arg2[%15, %9] : memref<128x128xf32>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
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
    byre.compute @PTXOp(%arg1, %113) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0_kernel"} : memref<2x128xi64>, memref<256xi1>
    byre.compute @AliasOp(%arg1, %112) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %114 = memref.alloc() : memref<256xui32>
    %115 = memref.alloc() : memref<256x1xi64>
    %116 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%arg0, %114, %115, %116) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1_kernel"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg4, %114, %105) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %117 = memref.alloc() : memref<256xui32>
    %118 = memref.alloc() : memref<256x1xi64>
    %119 = memref.alloc() : memref<256xi1>
    byre.compute @PTXOp(%107, %117, %118, %119) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2_kernel"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    byre.compute @IndexSelectOp(%arg5, %117, %104) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %120 = memref.alloc() : memref<128xui32>
    %121 = memref.alloc() : memref<128x1xi64>
    %122 = memref.alloc() : memref<128xi1>
    byre.compute @PTXOp(%106, %120, %121, %122) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [1 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3_kernel"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    byre.compute @IndexSelectOp(%arg6, %120, %103) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%105, %104, %103, %123) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4_kernel"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
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
    byre.compute @MatmulOp(%47, %arg4, %46) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    %124 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%46, %arg45, %arg46, %46, %124) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32, 3 : i32, 2 : i32], kernel_name = "Unknown5_kernel"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceMaxOp(%124, %45) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %125 = memref.alloc() : memref<256x30522xf32>
    %126 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%124, %45, %125, %126) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6_kernel"} : memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOp(%126, %44) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %127 = memref.alloc() : memref<256xf32>
    byre.compute @PTXOp(%44, %127) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7_kernel"} : memref<256xf32>, memref<256xf32>
    %128 = memref.alloc() : memref<256x30522xf32>
    %129 = memref.alloc() : memref<256x30522xf32>
    %130 = memref.alloc() : memref<256x30522xf32>
    %131 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%113, %112, %128, %125, %127, %129, %130, %131) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 1 : i32, 2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8_kernel"} : memref<256xi1>, memref<256xi64>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOp(%128, %43) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @ReduceSumOp(%128, %42) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %132 = memref.alloc() : memref<f32>
    byre.compute @PTXOp(%42, %132) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown9_kernel"} : memref<f32>, memref<f32>
    %133 = memref.alloc() : memref<256x30522xf32>
    byre.compute @PTXOp(%130, %132, %133) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 0 : i32, 2 : i32], kernel_name = "Unknown10_kernel"} : memref<256x30522xf32>, memref<f32>, memref<256x30522xf32>
    byre.compute @ReduceSumOp(%133, %41) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    byre.compute @ReduceSumOp(%129, %40) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%40, %43, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown11_kernel"} : memref<f32>, memref<f32>, memref<f32>
    %134 = memref.alloc() : memref<256x30522xf32>
    %135 = memref.alloc() : memref<2x128x30522xf32>
    byre.compute @PTXOp(%133, %131, %41, %134, %41, %135) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 2 : i32, 1 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12_kernel"} : memref<256x30522xf32>, memref<256x30522xf32>, memref<256xf32>, memref<256x30522xf32>, memref<256xf32>, memref<2x128x30522xf32>
    %136 = memref.alloc() : memref<30522x128xf32>
    byre.compute @MatmulOp(%47, %134, %136) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    byre.compute @MatmulOp(%134, %arg4, %39) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    byre.compute @AliasOp(%39, %38) {offset = 0 : i32} : memref<256x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%38, %53, %arg43, %49, %48, %37, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%37, %57, %arg41, %52, %51, %36, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%36, %54, %arg39, %56, %55, %35, %arg83, %arg84, %34) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%35, %61, %arg37, %33, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%33, %65, %arg35, %60, %59, %32, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %137 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%34, %32, %137) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown13_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
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
    byre.compute @PTXOp(%30, %21, %20, %19, %138) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%138, %77, %arg23, %79, %78, %18, %arg67, %arg68, %17) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%18, %84, %arg21, %16, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%16, %88, %arg19, %83, %82, %15, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %139 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @PTXOp(%17, %15, %139) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
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
    byre.compute @PTXOp(%13, %4, %3, %2, %140) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16_kernel"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%140, %123, %arg7, %101, %100, %1, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %141 = memref.alloc() : memref<256x128xf32>
    %142 = memref.alloc() : memref<256x128xf32>
    byre.compute @PTXOp(%116, %1, %141, %119, %142) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 3 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown17_kernel"} : memref<256xi1>, memref<2x128x128xf32>, memref<256x128xf32>, memref<256xi1>, memref<256x128xf32>
    byre.compute @IndexPutOp(%136, %115, %141, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOp(%110, %118, %142, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    byre.compute @ReduceSumOp(%1, %0) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %143 = memref.alloc() : memref<128x128xf32>
    byre.compute @PTXOp(%122, %0, %143) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18_kernel"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOp(%111, %121, %143, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOp(%135, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

