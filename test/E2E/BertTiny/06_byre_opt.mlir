// RUN: byteir-opt %s -byre-opt="append-arg-types" | FileCheck %s

// CHECK-LABEL: func.func @main
module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c16384 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15] : memref<128xi1>
        %17 = memref.load %arg1[%15, %9] : memref<128x128xf32>
        %18 = arith.select %16, %17, %cst : f32
        memref.store %18, %arg2[%15, %9] : memref<128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>, %arg3: memref<256x128xf32>, %arg4: memref<256x128xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
      %6 = arith.cmpi slt, %4, %c32768 : index
      scf.if %6 {
        %7 = arith.remsi %4, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = arith.select %8, %9, %7 : index
        %11 = arith.cmpi slt, %4, %c0 : index
        %12 = arith.subi %c-1, %4 : index
        %13 = arith.select %11, %12, %4 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = memref.load %arg0[%16] : memref<256xi1>
        %18 = memref.load %5[%16, %10] : memref<256x128xf32>
        %19 = arith.select %17, %18, %cst : f32
        memref.store %19, %arg3[%16, %10] : memref<256x128xf32>
        %20 = memref.load %arg2[%16] : memref<256xi1>
        %21 = arith.select %20, %18, %cst : f32
        memref.store %21, %arg4[%16, %10] : memref<256x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
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
    gpu.func @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>, %arg4: memref<2x128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
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
    gpu.func @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c32768 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c128 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c128 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c128 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = arith.remsi %15, %c128 : index
        %17 = arith.cmpi slt, %16, %c0 : index
        %18 = arith.addi %16, %c128 : index
        %19 = arith.select %17, %18, %16 : index
        %20 = arith.cmpi slt, %15, %c0 : index
        %21 = arith.subi %c-1, %15 : index
        %22 = arith.select %20, %21, %15 : index
        %23 = arith.divsi %22, %c128 : index
        %24 = arith.subi %c-1, %23 : index
        %25 = arith.select %20, %24, %23 : index
        %26 = memref.load %arg0[%25, %19, %9] : memref<2x128x128xf32>
        %27 = memref.load %arg1[%25, %19, %9] : memref<2x128x128xf32>
        %28 = arith.addf %26, %27 : f32
        memref.store %28, %arg2[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg2[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %18 = memref.load %arg0[%15] : memref<256xf32>
        %19 = arith.mulf %17, %18 : f32
        %20 = arith.subf %16, %19 : f32
        memref.store %20, %arg3[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown11(%arg0: memref<f32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg0[] : memref<f32>
        %18 = arith.divf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown10(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1 : index
      scf.if %5 {
        %6 = memref.load %arg0[] : memref<f32>
        %7 = arith.cmpf une, %6, %cst_0 : f32
        %8 = arith.select %7, %6, %cst : f32
        memref.store %8, %arg1[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) kernel {
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c1 : index
      scf.if %5 {
        %6 = memref.load %arg0[] : memref<f32>
        %7 = memref.load %arg1[] : memref<f32>
        %8 = arith.divf %6, %7 : f32
        memref.store %8, %arg2[] : memref<f32>
      }
      gpu.return
    }
    gpu.func @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>, %arg4: memref<256x30522xf32>, %arg5: memref<256x30522xf32>, %arg6: memref<256x30522xf32>, %arg7: memref<256x30522xf32>) kernel {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.alloca() : memref<256x30522xf32>
      %6 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %6 {
        %9 = arith.remsi %4, %c30522 : index
        %10 = arith.cmpi slt, %9, %c0 : index
        %11 = arith.addi %9, %c30522 : index
        %12 = arith.select %10, %11, %9 : index
        %13 = arith.cmpi slt, %4, %c0 : index
        %14 = arith.subi %c-1, %4 : index
        %15 = arith.select %13, %14, %4 : index
        %16 = arith.divsi %15, %c30522 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19 = memref.load %arg1[%18, %12] : memref<256x30522xf32>
        %20 = memref.load %arg0[%18] : memref<256xf32>
        %21 = arith.subf %19, %20 : f32
        memref.store %21, %5[%18, %12] : memref<256x30522xf32>
      }
      %7 = memref.alloca() : memref<256x30522xf32>
      scf.if %6 {
        %9 = arith.remsi %4, %c30522 : index
        %10 = arith.cmpi slt, %9, %c0 : index
        %11 = arith.addi %9, %c30522 : index
        %12 = arith.select %10, %11, %9 : index
        %13 = arith.cmpi slt, %4, %c0 : index
        %14 = arith.subi %c-1, %4 : index
        %15 = arith.select %13, %14, %4 : index
        %16 = arith.divsi %15, %c30522 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19 = memref.load %arg2[%18] : memref<256xi64>
        %20 = arith.index_cast %12 : index to i64
        %21 = arith.cmpi eq, %19, %20 : i64
        %22 = arith.select %21, %cst, %cst_0 : f32
        memref.store %22, %7[%18, %12] : memref<256x30522xf32>
      }
      %8 = memref.alloca() : memref<256x30522xf32>
      scf.if %6 {
        %9 = arith.remsi %4, %c30522 : index
        %10 = arith.cmpi slt, %9, %c0 : index
        %11 = arith.addi %9, %c30522 : index
        %12 = arith.select %10, %11, %9 : index
        %13 = arith.cmpi slt, %4, %c0 : index
        %14 = arith.subi %c-1, %4 : index
        %15 = arith.select %13, %14, %4 : index
        %16 = arith.divsi %15, %c30522 : index
        %17 = arith.subi %c-1, %16 : index
        %18 = arith.select %13, %17, %16 : index
        %19 = memref.load %7[%18, %12] : memref<256x30522xf32>
        %20 = arith.negf %19 : f32
        memref.store %20, %8[%18, %12] : memref<256x30522xf32>
        %21 = memref.load %arg3[%18] : memref<256xi1>
        %22 = arith.select %21, %cst, %cst_0 : f32
        %23 = arith.mulf %22, %19 : f32
        memref.store %23, %arg4[%18, %12] : memref<256x30522xf32>
        %24 = memref.load %8[%18, %12] : memref<256x30522xf32>
        %25 = memref.load %5[%18, %12] : memref<256x30522xf32>
        %26 = memref.load %arg4[%18, %12] : memref<256x30522xf32>
        %27 = arith.mulf %24, %25 : f32
        %28 = arith.cmpf une, %19, %cst : f32
        %29 = arith.select %28, %cst_0, %27 : f32
        %30 = arith.mulf %29, %26 : f32
        memref.store %30, %arg5[%18, %12] : memref<256x30522xf32>
        %31 = arith.mulf %24, %26 : f32
        memref.store %31, %arg6[%18, %12] : memref<256x30522xf32>
        %32 = math.exp %25 : f32
        memref.store %32, %arg7[%18, %12] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown7(%arg0: memref<256xf32>, %arg1: memref<256xf32>) kernel {
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
      scf.if %5 {
        %6 = memref.load %arg0[%4] : memref<256xf32>
        %7 = math.log %6 : f32
        memref.store %7, %arg1[%4] : memref<256xf32>
      }
      gpu.return
    }
    gpu.func @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg0[%15] : memref<256xf32>
        %18 = arith.subf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
        %19 = memref.load %arg2[%15, %9] : memref<256x30522xf32>
        %20 = math.exp %19 : f32
        memref.store %20, %arg3[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>, %arg2: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %5 {
        %6 = arith.remsi %4, %c30522 : index
        %7 = arith.cmpi slt, %6, %c0 : index
        %8 = arith.addi %6, %c30522 : index
        %9 = arith.select %7, %8, %6 : index
        %10 = arith.cmpi slt, %4, %c0 : index
        %11 = arith.subi %c-1, %4 : index
        %12 = arith.select %10, %11, %4 : index
        %13 = arith.divsi %12, %c30522 : index
        %14 = arith.subi %c-1, %13 : index
        %15 = arith.select %10, %14, %13 : index
        %16 = memref.load %arg0[%15, %9] : memref<256x30522xf32>
        %17 = memref.load %arg1[%9] : memref<30522xf32>
        %18 = arith.addf %16, %17 : f32
        memref.store %18, %arg2[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>, %arg3: memref<2x128x128xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c32768 = arith.constant 32768 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
      %6 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
      %7 = arith.cmpi slt, %4, %c32768 : index
      scf.if %7 {
        %8 = arith.remsi %4, %c128 : index
        %9 = arith.cmpi slt, %8, %c0 : index
        %10 = arith.addi %8, %c128 : index
        %11 = arith.select %9, %10, %8 : index
        %12 = arith.cmpi slt, %4, %c0 : index
        %13 = arith.subi %c-1, %4 : index
        %14 = arith.select %12, %13, %4 : index
        %15 = arith.divsi %14, %c128 : index
        %16 = arith.subi %c-1, %15 : index
        %17 = arith.select %12, %16, %15 : index
        %18 = arith.remsi %17, %c128 : index
        %19 = arith.cmpi slt, %18, %c0 : index
        %20 = arith.addi %18, %c128 : index
        %21 = arith.select %19, %20, %18 : index
        %22 = arith.cmpi slt, %17, %c0 : index
        %23 = arith.subi %c-1, %17 : index
        %24 = arith.select %22, %23, %17 : index
        %25 = arith.divsi %24, %c128 : index
        %26 = arith.subi %c-1, %25 : index
        %27 = arith.select %22, %26, %25 : index
        %28 = memref.load %6[%27, %21, %11] : memref<2x128x128xf32>
        %29 = memref.load %5[%27, %21, %11] : memref<2x128x128xf32>
        %30 = memref.load %arg2[%21, %11] : memref<128x128xf32>
        %31 = arith.addf %28, %29 : f32
        %32 = arith.addf %31, %30 : f32
        memref.store %32, %arg3[%27, %21, %11] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi1>) kernel {
      %c512_i64 = arith.constant 512 : i64
      %c0_i64 = arith.constant 0 : i64
      %c-1_i64 = arith.constant -1 : i64
      %c128 = arith.constant 128 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
      %6 = arith.cmpi slt, %4, %c128 : index
      scf.if %6 {
        %7 = memref.load %5[%4] : memref<128xi64>
        %8 = arith.trunci %7 : i64 to i32
        %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
        memref.store %9, %arg1[%4] : memref<128xui32>
        %10 = arith.addi %7, %c512_i64 : i64
        %11 = arith.cmpi slt, %7, %c0_i64 : i64
        %12 = arith.select %11, %10, %7 : i64
        memref.store %12, %arg2[%4] : memref<128xi64>
        %13 = arith.cmpi ne, %7, %c-1_i64 : i64
        memref.store %13, %arg3[%4] : memref<128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown2(%arg0: memref<128xi64>, %arg1: memref<2x128xui32>, %arg2: memref<2x128xi64>, %arg3: memref<2x128xi1>) kernel {
      %c2_i64 = arith.constant 2 : i64
      %c0_i64 = arith.constant 0 : i64
      %c-1_i64 = arith.constant -1 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.alloca() : memref<2x128xi64>
      %6 = arith.cmpi slt, %4, %c256 : index
      scf.if %6 {
        %7 = arith.remsi %4, %c128 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c128 : index
        %10 = arith.select %8, %9, %7 : index
        %11 = arith.cmpi slt, %4, %c0 : index
        %12 = arith.subi %c-1, %4 : index
        %13 = arith.select %11, %12, %4 : index
        %14 = arith.divsi %13, %c128 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = memref.load %arg0[%10] : memref<128xi64>
        memref.store %17, %5[%16, %10] : memref<2x128xi64>
        %18 = memref.load %5[%16, %10] : memref<2x128xi64>
        %19 = arith.trunci %18 : i64 to i32
        %20 = builtin.unrealized_conversion_cast %19 : i32 to ui32
        memref.store %20, %arg1[%16, %10] : memref<2x128xui32>
        %21 = arith.addi %18, %c2_i64 : i64
        %22 = arith.cmpi slt, %18, %c0_i64 : i64
        %23 = arith.select %22, %21, %18 : i64
        memref.store %23, %arg2[%16, %10] : memref<2x128xi64>
        %24 = arith.cmpi ne, %18, %c-1_i64 : i64
        memref.store %24, %arg3[%16, %10] : memref<2x128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<2x128xi64>, %arg1: memref<256xui32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) kernel {
      %c30522_i64 = arith.constant 30522 : i64
      %c0_i64 = arith.constant 0 : i64
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
      %6 = arith.cmpi slt, %4, %c256 : index
      scf.if %6 {
        %7 = memref.load %5[%4] : memref<256xi64>
        %8 = arith.trunci %7 : i64 to i32
        %9 = builtin.unrealized_conversion_cast %8 : i32 to ui32
        memref.store %9, %arg1[%4] : memref<256xui32>
        %10 = arith.addi %7, %c30522_i64 : i64
        %11 = arith.cmpi slt, %7, %c0_i64 : i64
        %12 = arith.select %11, %10, %7 : i64
        memref.store %12, %arg2[%4] : memref<256xi64>
        %13 = arith.cmpi ne, %7, %c0_i64 : i64
        memref.store %13, %arg3[%4] : memref<256xi1>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<256xi1>) kernel {
      %c-100_i64 = arith.constant -100 : i64
      %c256 = arith.constant 256 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
      %6 = arith.cmpi slt, %4, %c256 : index
      scf.if %6 {
        %7 = memref.load %5[%4] : memref<256xi64>
        %8 = arith.cmpi ne, %7, %c-100_i64 : i64
        memref.store %8, %arg1[%4] : memref<256xi1>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [2 : i32, 1 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, passthrough_arg = [1 : i32, 0 : i32]} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    %1 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    gpu.launch_func  @unified::@Unknown0 blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128xi64>, %0 : memref<256xi1>)
    return %1, %0 : memref<256xi64>, memref<256xi1>
  }
  func.func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown1", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<256xui32>
    gpu.launch_func  @unified::@Unknown1 blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128xi64>, %3 : memref<256xui32>, %1 : memref<256xi64>, %0 : memref<256xi1>)
    return %3, %2, %0 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown2", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128xi1>
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<2x128xi64>
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() {alignment = 128 : i64} : memref<2x128xui32>
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @unified::@Unknown2 blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xi64>, %5 : memref<2x128xui32>, %2 : memref<2x128xi64>, %0 : memref<2x128xi1>)
    return %6, %4, %1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], __byre__kernel_name = "Unknown3", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128xi1>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<128xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<128xui32>
    gpu.launch_func  @unified::@Unknown3 blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<1x128xi64>, %3 : memref<128xui32>, %1 : memref<128xi64>, %0 : memref<128xi1>)
    return %3, %2, %0 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func.func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], __byre__kernel_name = "Unknown4", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    gpu.launch_func  @unified::@Unknown4 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x128xf32>, %arg1 : memref<256x128xf32>, %arg2 : memref<128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
  }
  func.func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 61044 : i32, __byre__arg_ranks = [2 : i32, 1 : i32, 2 : i32], __byre__kernel_name = "Unknown5", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, passthrough_arg = [3 : i32, 2 : i32]} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c61044 = arith.constant 61044 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.expand_shape %0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    gpu.launch_func  @unified::@Unknown5 blocks in (%c61044, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256x30522xf32>, %arg1 : memref<30522xf32>, %0 : memref<256x30522xf32>)
    return %1, %0 : memref<2x128x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 61044 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown6", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c61044 = arith.constant 61044 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    gpu.launch_func  @unified::@Unknown6 blocks in (%c61044, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256x30522xf32>, %1 : memref<256x30522xf32>, %0 : memref<256x30522xf32>)
    return %1, %0 : memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 2 : i32, __byre__arg_ranks = [1 : i32, 1 : i32], __byre__kernel_name = "Unknown7", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    gpu.launch_func  @unified::@Unknown7 blocks in (%c2, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %0 : memref<256xf32>)
    return %0 : memref<256xf32>
  }
  func.func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 61044 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown8", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c61044 = arith.constant 61044 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %2 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %3 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    gpu.launch_func  @unified::@Unknown8 blocks in (%c61044, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256x30522xf32>, %arg2 : memref<256xi64>, %arg3 : memref<256xi1>, %3 : memref<256x30522xf32>, %2 : memref<256x30522xf32>, %1 : memref<256x30522xf32>, %0 : memref<256x30522xf32>)
    return %3, %2, %1, %0 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32, 0 : i32], __byre__kernel_name = "Unknown9", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
    gpu.launch_func  @unified::@Unknown9 blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<f32>, %arg1 : memref<f32>, %0 : memref<f32>)
    return %0 : memref<f32>
  }
  func.func private @Unknown10(%arg0: memref<f32>) -> memref<f32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 1 : i32, __byre__arg_ranks = [0 : i32, 0 : i32], __byre__kernel_name = "Unknown10", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<f32>
    gpu.launch_func  @unified::@Unknown10 blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<f32>, %0 : memref<f32>)
    return %0 : memref<f32>
  }
  func.func private @Unknown11(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 61044 : i32, __byre__arg_ranks = [0 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown11", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c61044 = arith.constant 61044 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    gpu.launch_func  @unified::@Unknown11 blocks in (%c61044, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<f32>, %arg1 : memref<256x30522xf32>, %0 : memref<256x30522xf32>)
    return %0 : memref<256x30522xf32>
  }
  func.func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 61044 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown12", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, passthrough_arg = [4 : i32, 3 : i32]} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c61044 = arith.constant 61044 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    %1 = memref.expand_shape %0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    gpu.launch_func  @unified::@Unknown12 blocks in (%c61044, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xf32>, %arg1 : memref<256x30522xf32>, %arg2 : memref<256x30522xf32>, %0 : memref<256x30522xf32>)
    return %0, %1 : memref<256x30522xf32>, memref<2x128x30522xf32>
  }
  func.func private @MatmulOp13(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func.func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown14", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    gpu.launch_func  @unified::@Unknown14 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
  }
  func.func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown15", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    gpu.launch_func  @unified::@Unknown15 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
  }
  func.func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown16", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    gpu.launch_func  @unified::@Unknown16 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
  }
  func.func private @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown17", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    gpu.launch_func  @unified::@Unknown17 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
  }
  func.func private @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 256 : i32, __byre__arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown18", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    gpu.launch_func  @unified::@Unknown18 blocks in (%c256, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<256xi1>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<256xi1>, %1 : memref<256x128xf32>, %0 : memref<256x128xf32>)
    return %1, %0 : memref<256x128xf32>, memref<256x128xf32>
  }
  func.func private @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byre__BlockSize.x = 128 : i32, __byre__GridSize.x = 128 : i32, __byre__arg_ranks = [1 : i32, 2 : i32, 2 : i32], __byre__kernel_name = "Unknown19", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    gpu.launch_func  @unified::@Unknown19 blocks in (%c128, %c1, %c1) threads in (%c128, %c1, %c1) args(%arg0 : memref<128xi1>, %arg1 : memref<128x128xf32>, %0 : memref<128x128xf32>)
    return %0 : memref<128x128xf32>
  }
  func.func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %1 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %2 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.constant"(%2) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    %3 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%3) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    %4 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%4) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %5 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %5) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %6 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%5, %6) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %7 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg3, %7) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %8:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %9:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %10 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %9#0, %10) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %11:3 = call @Unknown2(%6) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %12 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg5, %11#0, %12) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %13:3 = call @Unknown3(%7) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %14 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg6, %13#0, %14) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %15 = call @Unknown4(%10, %12, %14) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    %16 = memref.alloc() : memref<2x128x128xf32>
    %17 = memref.alloc() : memref<256xf32>
    %18 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%15, %arg7, %arg8, %16, %17, %18) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %19 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg9, %arg10, %19) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %20 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg11, %arg12, %20) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %21 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%19, %20, %21) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %22 = memref.alloc() : memref<2x2x128x128xf32>
    %23 = memref.alloc() : memref<2x2x128x128xf32>
    %24 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%21, %2, %22, %23, %24) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %25 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%16, %arg13, %arg14, %25) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %26 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%22, %25, %26) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %27 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%26, %27) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %28 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%27, %28) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %29 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%28, %arg15, %arg16, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %30 = memref.alloc() : memref<2x128x128xf32>
    %31 = memref.alloc() : memref<256xf32>
    %32 = memref.alloc() : memref<256xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%29, %arg17, %arg18, %16, %30, %31, %32, %33) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %34 = memref.alloc() : memref<2x128x512xf32>
    %35 = memref.alloc() : memref<2x128x512xf32>
    %36 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%30, %arg19, %arg20, %34, %35, %36) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %37 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%34, %arg21, %arg22, %37) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %38 = memref.alloc() : memref<2x128x128xf32>
    %39 = memref.alloc() : memref<256xf32>
    %40 = memref.alloc() : memref<256xf32>
    %41 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%37, %arg23, %arg24, %30, %38, %39, %40, %41) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %42 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg25, %arg26, %42) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %43 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg27, %arg28, %43) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %44 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%42, %43, %44) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %45 = memref.alloc() : memref<2x2x128x128xf32>
    %46 = memref.alloc() : memref<2x2x128x128xf32>
    %47 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%44, %2, %45, %46, %47) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %48 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%38, %arg29, %arg30, %48) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %49 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%45, %48, %49) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %50 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%49, %50) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %51 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%50, %51) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %52 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%51, %arg31, %arg32, %52) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %53 = memref.alloc() : memref<2x128x128xf32>
    %54 = memref.alloc() : memref<256xf32>
    %55 = memref.alloc() : memref<256xf32>
    %56 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%52, %arg33, %arg34, %38, %53, %54, %55, %56) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %57 = memref.alloc() : memref<2x128x512xf32>
    %58 = memref.alloc() : memref<2x128x512xf32>
    %59 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%53, %arg35, %arg36, %57, %58, %59) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %60 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%57, %arg37, %arg38, %60) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %61 = memref.alloc() : memref<2x128x128xf32>
    %62 = memref.alloc() : memref<256xf32>
    %63 = memref.alloc() : memref<256xf32>
    %64 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%60, %arg39, %arg40, %53, %61, %62, %63, %64) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %65 = memref.alloc() : memref<2x128x128xf32>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%61, %arg41, %arg42, %65, %66, %67) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %68 = memref.alloc() : memref<2x128x128xf32>
    %69 = memref.alloc() : memref<256xf32>
    %70 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%65, %arg43, %arg44, %68, %69, %70) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %71 = memref.alloc() : memref<256x128xf32>
    "lmhlo.reshape"(%68, %71) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    %72 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.dot"(%71, %arg4, %72) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %73:2 = call @Unknown5(%72, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    %74 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%73#1, %3, %74) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %75:2 = call @Unknown6(%74, %73#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    %76 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%75#1, %4, %76) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %77 = call @Unknown7(%76) : (memref<256xf32>) -> memref<256xf32>
    %78:4 = call @Unknown8(%77, %75#0, %8#0, %8#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    %79 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#1, %4, %79) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %80 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#0, %4, %80) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %81 = call @Unknown9(%79, %80) : (memref<f32>, memref<f32>) -> memref<f32>
    %82 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%78#0, %4, %82) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %83 = call @Unknown10(%82) : (memref<f32>) -> memref<f32>
    %84 = call @Unknown11(%83, %78#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    %85 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%84, %4, %85) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %86:2 = call @Unknown12(%85, %78#3, %84) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %87 = call @MatmulOp13(%71, %86#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    %88 = memref.alloc() : memref<256x128xf32>
    "lmhlo.dot"(%86#0, %arg4, %88) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    %89 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%88, %89) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<128xf32>
    %92 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%89, %65, %arg43, %69, %70, %90, %91, %92) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %93 = memref.alloc() : memref<2x128x128xf32>
    %94 = memref.alloc() : memref<128x128xf32>
    %95 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%90, %61, %arg41, %66, %67, %93, %94, %95) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %96 = memref.alloc() : memref<2x128x128xf32>
    %97 = memref.alloc() : memref<128xf32>
    %98 = memref.alloc() : memref<128xf32>
    %99 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%93, %64, %arg39, %62, %63, %96, %97, %98, %99) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %100 = memref.alloc() : memref<2x128x512xf32>
    %101 = memref.alloc() : memref<128x512xf32>
    %102 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%96, %57, %arg37, %100, %101, %102) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %103 = memref.alloc() : memref<2x128x128xf32>
    %104 = memref.alloc() : memref<512x128xf32>
    %105 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%100, %53, %arg35, %58, %59, %103, %104, %105) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %106 = call @Unknown14(%99, %103) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %107 = memref.alloc() : memref<2x128x128xf32>
    %108 = memref.alloc() : memref<128xf32>
    %109 = memref.alloc() : memref<128xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%106, %56, %arg33, %54, %55, %107, %108, %109, %110) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %111 = memref.alloc() : memref<2x128x128xf32>
    %112 = memref.alloc() : memref<128x128xf32>
    %113 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%107, %51, %arg31, %111, %112, %113) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %114 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%111, %114) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %115 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%114, %115) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%115, %45, %48, %116, %117) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %118 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%116, %45, %47, %118) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %119 = memref.alloc() : memref<2x2x128x64xf32>
    %120 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%118, %42, %43, %119, %120) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %121 = memref.alloc() : memref<2x128x128xf32>
    %122 = memref.alloc() : memref<128x128xf32>
    %123 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%119, %38, %arg25, %121, %122, %123) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<128x128xf32>
    %126 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%117, %38, %arg29, %124, %125, %126) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %127 = memref.alloc() : memref<2x128x128xf32>
    %128 = memref.alloc() : memref<128x128xf32>
    %129 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%120, %38, %arg27, %127, %128, %129) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %130 = call @Unknown15(%110, %121, %124, %127) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %131 = memref.alloc() : memref<2x128x128xf32>
    %132 = memref.alloc() : memref<128xf32>
    %133 = memref.alloc() : memref<128xf32>
    %134 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%130, %41, %arg23, %39, %40, %131, %132, %133, %134) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %135 = memref.alloc() : memref<2x128x512xf32>
    %136 = memref.alloc() : memref<128x512xf32>
    %137 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%131, %34, %arg21, %135, %136, %137) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %138 = memref.alloc() : memref<2x128x128xf32>
    %139 = memref.alloc() : memref<512x128xf32>
    %140 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%135, %30, %arg19, %35, %36, %138, %139, %140) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %141 = call @Unknown16(%134, %138) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %142 = memref.alloc() : memref<2x128x128xf32>
    %143 = memref.alloc() : memref<128xf32>
    %144 = memref.alloc() : memref<128xf32>
    %145 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%141, %33, %arg17, %31, %32, %142, %143, %144, %145) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %146 = memref.alloc() : memref<2x128x128xf32>
    %147 = memref.alloc() : memref<128x128xf32>
    %148 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%142, %28, %arg15, %146, %147, %148) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %149 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%146, %149) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %150 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%149, %150) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %151 = memref.alloc() : memref<2x2x128x128xf32>
    %152 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%150, %22, %25, %151, %152) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %153 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%151, %22, %24, %153) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %154 = memref.alloc() : memref<2x2x128x64xf32>
    %155 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%153, %19, %20, %154, %155) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %156 = memref.alloc() : memref<2x128x128xf32>
    %157 = memref.alloc() : memref<128x128xf32>
    %158 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%154, %16, %arg9, %156, %157, %158) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %159 = memref.alloc() : memref<2x128x128xf32>
    %160 = memref.alloc() : memref<128x128xf32>
    %161 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%152, %16, %arg13, %159, %160, %161) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %162 = memref.alloc() : memref<2x128x128xf32>
    %163 = memref.alloc() : memref<128x128xf32>
    %164 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%155, %16, %arg11, %162, %163, %164) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %165 = call @Unknown17(%145, %156, %159, %162) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %166 = memref.alloc() : memref<2x128x128xf32>
    %167 = memref.alloc() : memref<128xf32>
    %168 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%165, %15, %arg7, %17, %18, %166, %167, %168) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %169:2 = call @Unknown18(%9#2, %166, %11#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %170 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%87, %9#1, %169#0, %170) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %171 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%1, %11#1, %169#1, %171) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %172 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%166, %4, %172) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %173 = call @Unknown19(%13#2, %172) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    %174 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%0, %13#1, %173, %174) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %176 = mhlo.add %arg46, %arg47 : tensor<f32>
      "mhlo.return"(%176) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    %175 = memref.alloc() : memref<30522xf32>
    "lmhlo.reduce"(%86#1, %4, %175) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %73#0, %81, %170, %171, %174, %167, %168, %157, %158, %163, %164, %160, %161, %147, %148, %143, %144, %139, %140, %136, %137, %132, %133, %122, %123, %128, %129, %125, %126, %112, %113, %108, %109, %104, %105, %101, %102, %97, %98, %94, %95, %91, %92, %175 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

