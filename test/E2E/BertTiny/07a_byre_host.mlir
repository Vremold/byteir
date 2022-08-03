// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cuda" | FileCheck %s

// CHECK-LABEL: func.func @main
module attributes {byre.container_module, gpu.container_module} {
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
  func.func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<0xi8>
    %1 = memref.alloc() : memref<0xi8>
    %2 = memref.alloc() : memref<0xi8>
    %3 = memref.alloc() : memref<16xi8>
    %4 = memref.alloc() : memref<32xi8>
    %5 = memref.alloc() : memref<32xi8>
    %6 = memref.alloc() : memref<32xi8>
    %7 = memref.alloc() : memref<1024xi8>
    %8 = memref.alloc() : memref<1024xi8>
    %9 = memref.alloc() : memref<1024xi8>
    %10 = memref.alloc() : memref<1024xi8>
    %11 = memref.alloc() : memref<1024xi8>
    %12 = memref.alloc() : memref<1024xi8>
    %13 = memref.alloc() : memref<1024xi8>
    %14 = memref.alloc() : memref<1024xi8>
    %15 = memref.alloc() : memref<1024xi8>
    %16 = memref.alloc() : memref<1024xi8>
    %17 = memref.alloc() : memref<1024xi8>
    %18 = memref.alloc() : memref<1024xi8>
    %19 = memref.alloc() : memref<1024xi8>
    %20 = memref.alloc() : memref<1024xi8>
    %21 = memref.alloc() : memref<2048xi8>
    %22 = memref.alloc() : memref<2048xi8>
    %23 = memref.alloc() : memref<65536xi8>
    %24 = memref.alloc() : memref<65536xi8>
    %25 = memref.alloc() : memref<131072xi8>
    %26 = memref.alloc() : memref<131072xi8>
    %27 = memref.alloc() : memref<131072xi8>
    %28 = memref.alloc() : memref<131072xi8>
    %29 = memref.alloc() : memref<131072xi8>
    %30 = memref.alloc() : memref<131072xi8>
    %31 = memref.alloc() : memref<131072xi8>
    %32 = memref.alloc() : memref<131072xi8>
    %33 = memref.alloc() : memref<131072xi8>
    %34 = memref.alloc() : memref<131072xi8>
    %35 = memref.alloc() : memref<131072xi8>
    %36 = memref.alloc() : memref<131072xi8>
    %37 = memref.alloc() : memref<131072xi8>
    %38 = memref.alloc() : memref<131072xi8>
    %39 = memref.alloc() : memref<131072xi8>
    %40 = memref.alloc() : memref<131072xi8>
    %41 = memref.alloc() : memref<131072xi8>
    %42 = memref.alloc() : memref<131072xi8>
    %43 = memref.alloc() : memref<131072xi8>
    %44 = memref.alloc() : memref<131072xi8>
    %45 = memref.alloc() : memref<131072xi8>
    %46 = memref.alloc() : memref<262144xi8>
    %47 = memref.alloc() : memref<262144xi8>
    %48 = memref.alloc() : memref<524288xi8>
    %49 = memref.alloc() : memref<524288xi8>
    %50 = memref.alloc() : memref<524288xi8>
    %51 = memref.alloc() : memref<524288xi8>
    %52 = memref.alloc() : memref<31254528xi8>
    %53 = memref.alloc() : memref<31254528xi8>
    %54 = memref.alloc() : memref<31254528xi8>
    %55 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%55) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    %56 = memref.alloc() : memref<2x128xf32>
    byre.compute @FillOp(%56) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    %57 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @FillOp(%57) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32>
    %58 = memref.alloc() : memref<128xi64>
    byre.compute @AliasOp(%arg2, %58) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<128xi64>
    %59 = memref.alloc() : memref<1x128xi64>
    byre.compute @AliasOp(%arg3, %59) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %60 = memref.alloc() : memref<256xi64>
    byre.compute @AliasOp(%arg1, %60) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %61 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%6, %61) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%arg1, %61) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0"} : memref<2x128xi64>, memref<256xi1>
    byre.compute @AliasOp(%arg1, %60) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %62 = memref.alloc() : memref<256xui32>
    byre.compute @AliasOp(%54, %62) {offset = 0 : i32} : memref<31254528xi8>, memref<256xui32>
    %63 = memref.alloc() : memref<256x1xi64>
    byre.compute @AliasOp(%21, %63) {offset = 0 : index} : memref<2048xi8>, memref<256x1xi64>
    %64 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%4, %64) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%arg0, %62, %63, %64) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %65 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%54, %65) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %62, %65) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %66 = memref.alloc() : memref<256xui32>
    byre.compute @AliasOp(%54, %66) {offset = 131072 : index} : memref<31254528xi8>, memref<256xui32>
    %67 = memref.alloc() : memref<256x1xi64>
    byre.compute @AliasOp(%22, %67) {offset = 0 : index} : memref<2048xi8>, memref<256x1xi64>
    %68 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%5, %68) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%58, %66, %67, %68) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %69 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%54, %69) {offset = 0 : i32} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %66, %69) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %70 = memref.alloc() : memref<128xui32>
    byre.compute @AliasOp(%54, %70) {offset = 131072 : index} : memref<31254528xi8>, memref<128xui32>
    %71 = memref.alloc() : memref<128x1xi64>
    byre.compute @AliasOp(%14, %71) {offset = 0 : index} : memref<1024xi8>, memref<128x1xi64>
    %72 = memref.alloc() : memref<128xi1>
    byre.compute @AliasOp(%3, %72) {offset = 0 : index} : memref<16xi8>, memref<128xi1>
    byre.compute @PTXOp(%59, %70, %71, %72) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    %73 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%54, %73) {offset = 131072 : i32} : memref<31254528xi8>, memref<128x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %70, %73) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %74 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%33, %74) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%65, %69, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    %75 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%40, %75) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %76 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%18, %76) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %77 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%17, %77) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @ftv4.layernorm(%74, %arg7, %arg8, %75, %76, %77) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %78 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%35, %78) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%75, %arg9, %arg10, %78) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %79 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%36, %79) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%75, %arg11, %arg12, %79) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %80 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %80) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%78, %79, %80) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %81 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%46, %81) {offset = 0 : index} : memref<262144xi8>, memref<2x2x128x128xf32>
    %82 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %82) {offset = 0 : i32} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %83 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @AliasOp(%23, %83) {offset = 0 : index} : memref<65536xi8>, memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%80, %57, %81, %82, %83) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %84 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%37, %84) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%75, %arg13, %arg14, %84) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %85 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %85) {offset = 0 : i32} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%81, %84, %85) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %86 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%38, %86) {offset = 0 : index} : memref<131072xi8>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%85, %86) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %87 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%38, %87) {offset = 0 : i32} : memref<131072xi8>, memref<2x128x128xf32>
    %88 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %88) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%87, %arg15, %arg16, %88) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %89 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%39, %89) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %90 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%16, %90) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %91 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%15, %91) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %92 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%25, %92) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%88, %arg17, %arg18, %75, %89, %90, %91, %92) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %93 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%49, %93) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %94 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%50, %94) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %95 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%2, %95) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%89, %arg19, %arg20, %93, %94, %95) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %96 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %96) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%93, %arg21, %arg22, %96) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %97 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%26, %97) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %98 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%13, %98) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %99 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%12, %99) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %100 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%27, %100) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%96, %arg23, %arg24, %89, %97, %98, %99, %100) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %101 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%28, %101) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%97, %arg25, %arg26, %101) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %102 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%32, %102) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%97, %arg27, %arg28, %102) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %103 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %103) {offset = 0 : i32} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%101, %102, %103) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %104 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%47, %104) {offset = 0 : index} : memref<262144xi8>, memref<2x2x128x128xf32>
    %105 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %105) {offset = 262144 : i32} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %106 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @AliasOp(%24, %106) {offset = 0 : index} : memref<65536xi8>, memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%103, %57, %104, %105, %106) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %107 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%29, %107) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%97, %arg29, %arg30, %107) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %108 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %108) {offset = 0 : i32} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%104, %107, %108) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %109 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%30, %109) {offset = 0 : index} : memref<131072xi8>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%108, %109) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %110 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%30, %110) {offset = 0 : i32} : memref<131072xi8>, memref<2x128x128xf32>
    %111 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %111) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%110, %arg31, %arg32, %111) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %112 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%31, %112) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %113 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%11, %113) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %114 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%10, %114) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %115 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%34, %115) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%111, %arg33, %arg34, %97, %112, %113, %114, %115) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %116 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%48, %116) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %117 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%51, %117) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %118 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%1, %118) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%112, %arg35, %arg36, %116, %117, %118) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %119 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %119) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%116, %arg37, %arg38, %119) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %120 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%43, %120) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %121 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%19, %121) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %122 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%9, %122) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %123 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%44, %123) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%119, %arg39, %arg40, %112, %120, %121, %122, %123) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %124 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%42, %124) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %125 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%45, %125) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %126 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%0, %126) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%120, %arg41, %arg42, %124, %125, %126) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    %127 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%41, %127) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %128 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%20, %128) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %129 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%8, %129) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @ftv4.layernorm(%124, %arg43, %arg44, %127, %128, %129) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %130 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%41, %130) {offset = 0 : i32} : memref<131072xi8>, memref<256x128xf32>
    %131 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %131) {offset = 0 : i32} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @MatmulOpf32f32f32(%130, %arg4, %131) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    %132 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%arg46, %132) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256x30522xf32>
    byre.compute @PTXOp(%131, %arg45, %arg46) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32], kernel_name = "Unknown5"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>
    byre.compute @AliasOp(%arg46, %132) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256x30522xf32>
    %133 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%53, %133) {offset = 0 : index} : memref<31254528xi8>, memref<256xf32>
    byre.compute @ReduceMaxOpf32f32(%132, %133) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %134 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %134) {offset = 0 : i32} : memref<31254528xi8>, memref<256x30522xf32>
    %135 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %135) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%133, %132, %134, %135) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %136 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%arg46, %136) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%135, %136) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %137 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%7, %137) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @PTXOp(%136, %137) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7"} : memref<256xf32>, memref<256xf32>
    %138 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %138) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %139 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%arg46, %139) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256x30522xf32>
    %140 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%53, %140) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %141 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%52, %141) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%137, %134, %60, %61, %138, %139, %140, %141) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8"} : memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %142 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%54, %142) {offset = 0 : i32} : memref<31254528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%139, %142) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %143 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%54, %143) {offset = 4 : i32} : memref<31254528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%138, %143) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%142, %143, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown9"} : memref<f32>, memref<f32>, memref<f32>
    %144 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%54, %144) {offset = 0 : i32} : memref<31254528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%138, %144) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %145 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%54, %145) {offset = 0 : index} : memref<31254528xi8>, memref<f32>
    byre.compute @PTXOp(%144, %145) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown10"} : memref<f32>, memref<f32>
    %146 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %146) {offset = 0 : i32} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%145, %140, %146) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown11"} : memref<f32>, memref<256x30522xf32>, memref<256x30522xf32>
    %147 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%arg46, %147) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%146, %147) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %148 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %148) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %149 = memref.alloc() : memref<2x128x30522xf32>
    byre.compute @AliasOp(%54, %149) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x30522xf32>
    byre.compute @PTXOp(%147, %141, %146, %148) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown12"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @AliasOp(%54, %149) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x30522xf32>
    %150 = memref.alloc() : memref<30522x128xf32>
    byre.compute @AliasOp(%arg46, %150) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%130, %148, %150) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    %151 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%53, %151) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%148, %arg4, %151) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    %152 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%53, %152) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    %153 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %153) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%152, %124, %arg43, %128, %129, %153, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %154 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %154) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%153, %120, %arg41, %125, %126, %154, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %155 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %155) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    %156 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%53, %156) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%154, %123, %arg39, %121, %122, %155, %arg83, %arg84, %156) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %157 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%54, %157) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%155, %116, %arg37, %157, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %158 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %158) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%157, %112, %arg35, %117, %118, %158, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %159 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %159) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%156, %158, %159) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %160 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %160) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    %161 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%52, %161) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%159, %115, %arg33, %113, %114, %160, %arg77, %arg78, %161) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %162 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %162) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%160, %110, %arg31, %162, %arg75, %arg76) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %163 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%54, %163) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x2x64xf32>
    %164 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%arg46, %164) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%163, %164) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %165 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %165) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %166 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%53, %166) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%164, %104, %107, %165, %166) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %167 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%arg46, %167) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%165, %104, %106, %167) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %168 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %168) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    %169 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%52, %169) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%167, %101, %102, %168, %169) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %170 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %170) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%168, %97, %arg25, %170, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %171 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %171) {arg_alias, offset = 15758336 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%166, %97, %arg29, %171, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %172 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %172) {arg_alias, offset = 15889408 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%169, %97, %arg27, %172, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %173 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %173) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%161, %170, %171, %172, %173) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %174 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %174) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    %175 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%53, %175) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%173, %100, %arg23, %98, %99, %174, %arg67, %arg68, %175) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %176 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%54, %176) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%174, %93, %arg21, %176, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %177 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %177) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%176, %89, %arg19, %94, %95, %177, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %178 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %178) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%175, %177, %178) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %179 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %179) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    %180 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%53, %180) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%178, %92, %arg17, %90, %91, %179, %arg61, %arg62, %180) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %181 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %181) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%179, %87, %arg15, %181, %arg59, %arg60) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %182 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%54, %182) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x2x64xf32>
    %183 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%arg46, %183) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%182, %183) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %184 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%54, %184) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %185 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%53, %185) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%183, %81, %84, %184, %185) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %186 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%arg46, %186) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%184, %81, %83, %186) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %187 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %187) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    %188 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %188) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%186, %78, %79, %187, %188) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %189 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %189) {arg_alias, offset = 15627264 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%187, %75, %arg9, %189, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %190 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %190) {arg_alias, offset = 15758336 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%185, %75, %arg13, %190, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %191 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%arg46, %191) {arg_alias, offset = 15889408 : i32} : memref<2x128x30522xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%188, %75, %arg11, %191, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %192 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %192) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%180, %189, %190, %191, %192) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %193 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%53, %193) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%192, %74, %arg7, %76, %77, %193, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %194 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%54, %194) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    %195 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%53, %195) {offset = 131072 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @PTXOp(%64, %193, %68, %194, %195) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18"} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%150, %63, %194, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%56, %67, %195, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    %196 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%arg46, %196) {arg_alias, offset = 0 : i32} : memref<2x128x30522xf32>, memref<128x128xf32>
    byre.compute @ReduceSumOpf32f32(%193, %196) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %197 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%54, %197) {offset = 0 : index} : memref<31254528xi8>, memref<128x128xf32>
    byre.compute @PTXOp(%72, %196, %197) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%55, %71, %197, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%149, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

