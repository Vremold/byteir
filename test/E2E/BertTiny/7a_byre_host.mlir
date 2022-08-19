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
    %55 = memref.alloc() : memref<31254528xi8>
    %56 = memref.alloc() : memref<31254528xi8>
    %57 = memref.alloc() : memref<512x128xf32>
    byre.compute @FillOp(%57) {value = dense<0.000000e+00> : tensor<512x128xf32>} : memref<512x128xf32>
    %58 = memref.alloc() : memref<2x128xf32>
    byre.compute @FillOp(%58) {value = dense<0.000000e+00> : tensor<2x128xf32>} : memref<2x128xf32>
    %59 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @FillOp(%59) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : memref<2x128x128xf32>
    %60 = "byre.alias"(%arg2) {offset = 0 : i64} : (memref<1x512xi64>) -> memref<128xi64>
    %61 = "byre.alias"(%arg3) {offset = 0 : i64} : (memref<1x512xi64>) -> memref<1x128xi64>
    %62 = "byre.alias"(%6) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%arg1, %62) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0"} : memref<2x128xi64>, memref<256xi1>
    %63 = "byre.alias"(%arg1) {offset = 0 : i64} : (memref<2x128xi64>) -> memref<256xi64>
    %64 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256xui32>
    %65 = "byre.alias"(%22) {offset = 0 : i64} : (memref<2048xi8>) -> memref<256x1xi64>
    %66 = "byre.alias"(%4) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%arg0, %64, %65, %66) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %67 = "byre.alias"(%55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %64, %67) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %68 = "byre.alias"(%55) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<256xui32>
    %69 = "byre.alias"(%21) {offset = 0 : i64} : (memref<2048xi8>) -> memref<256x1xi64>
    %70 = "byre.alias"(%5) {offset = 0 : i64} : (memref<32xi8>) -> memref<256xi1>
    byre.compute @PTXOp(%60, %68, %69, %70) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %71 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %68, %71) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %72 = "byre.alias"(%55) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<128xui32>
    %73 = "byre.alias"(%19) {offset = 0 : i64} : (memref<1024xi8>) -> memref<128x1xi64>
    %74 = "byre.alias"(%3) {offset = 0 : i64} : (memref<16xi8>) -> memref<128xi1>
    byre.compute @PTXOp(%61, %72, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    %75 = "byre.alias"(%56) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %72, %75) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %76 = "byre.alias"(%35) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @PTXOp(%67, %71, %75, %76) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    %77 = "byre.alias"(%36) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %78 = "byre.alias"(%18) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %79 = "byre.alias"(%17) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @ftv4.layernorm(%76, %arg7, %arg8, %77, %78, %79) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %80 = "byre.alias"(%37) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg9, %arg10, %80) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %81 = "byre.alias"(%38) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg11, %arg12, %81) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %82 = "byre.alias"(%55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%80, %81, %82) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %83 = "byre.alias"(%47) {offset = 0 : i64} : (memref<262144xi8>) -> memref<2x2x128x128xf32>
    %84 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    %85 = "byre.alias"(%24) {offset = 0 : i64} : (memref<65536xi8>) -> memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%82, %59, %83, %84, %85) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %86 = "byre.alias"(%39) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg13, %arg14, %86) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %87 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%83, %86, %87) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %88 = "byre.alias"(%40) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%87, %88) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %89 = "byre.alias"(%40) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %90 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear(%89, %arg15, %arg16, %90) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %91 = "byre.alias"(%34) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %92 = "byre.alias"(%16) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %93 = "byre.alias"(%15) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %94 = "byre.alias"(%25) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%90, %arg17, %arg18, %77, %91, %92, %93, %94) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %95 = "byre.alias"(%49) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %96 = "byre.alias"(%50) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %97 = "byre.alias"(%2) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%91, %arg19, %arg20, %95, %96, %97) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%95, %arg21, %arg22, %90) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %98 = "byre.alias"(%26) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %99 = "byre.alias"(%14) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %100 = "byre.alias"(%13) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %101 = "byre.alias"(%27) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%90, %arg23, %arg24, %91, %98, %99, %100, %101) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %102 = "byre.alias"(%28) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%98, %arg25, %arg26, %102) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %103 = "byre.alias"(%29) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%98, %arg27, %arg28, %103) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%102, %103, %82) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %104 = "byre.alias"(%46) {offset = 0 : i64} : (memref<262144xi8>) -> memref<2x2x128x128xf32>
    %105 = "byre.alias"(%23) {offset = 0 : i64} : (memref<65536xi8>) -> memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%82, %59, %104, %84, %105) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %106 = "byre.alias"(%30) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%98, %arg29, %arg30, %106) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%104, %106, %87) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %107 = "byre.alias"(%31) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%87, %107) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %108 = "byre.alias"(%31) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear(%108, %arg31, %arg32, %90) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %109 = "byre.alias"(%32) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %110 = "byre.alias"(%12) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %111 = "byre.alias"(%20) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %112 = "byre.alias"(%33) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%90, %arg33, %arg34, %98, %109, %110, %111, %112) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %113 = "byre.alias"(%51) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %114 = "byre.alias"(%48) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x512xf32>
    %115 = "byre.alias"(%1) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%109, %arg35, %arg36, %113, %114, %115) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    byre.compute @ftv4.linear(%113, %arg37, %arg38, %90) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %116 = "byre.alias"(%42) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %117 = "byre.alias"(%11) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %118 = "byre.alias"(%10) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %119 = "byre.alias"(%43) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%90, %arg39, %arg40, %109, %116, %117, %118, %119) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %120 = "byre.alias"(%44) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %121 = "byre.alias"(%45) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %122 = "byre.alias"(%0) {offset = 0 : i64} : (memref<0xi8>) -> memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%116, %arg41, %arg42, %120, %121, %122) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    %123 = "byre.alias"(%41) {offset = 0 : i64} : (memref<131072xi8>) -> memref<2x128x128xf32>
    %124 = "byre.alias"(%9) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    %125 = "byre.alias"(%8) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @ftv4.layernorm(%120, %arg43, %arg44, %123, %124, %125) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %126 = "byre.alias"(%41) {offset = 0 : i64} : (memref<131072xi8>) -> memref<256x128xf32>
    %127 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @MatmulOpf32f32f32(%126, %arg4, %127) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    byre.compute @PTXOp(%127, %arg45, %arg46) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 2 : i32], kernel_name = "Unknown5"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>
    %128 = "byre.alias"(%arg46) {offset = 0 : i64} : (memref<2x128x30522xf32>) -> memref<256x30522xf32>
    %129 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256xf32>
    byre.compute @ReduceMaxOpf32f32(%128, %129) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %130 = "byre.alias"(%55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @PTXOp(%129, %128, %127, %130) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%130, %129) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %131 = "byre.alias"(%7) {offset = 0 : i64} : (memref<1024xi8>) -> memref<256xf32>
    byre.compute @PTXOp(%129, %131) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7"} : memref<256xf32>, memref<256xf32>
    %132 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    %133 = "byre.alias"(%53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    %134 = "byre.alias"(%52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<256x30522xf32>
    byre.compute @PTXOp(%131, %127, %63, %62, %130, %132, %133, %134) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8"} : memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %135 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<f32>
    byre.compute @ReduceSumOpf32f32(%132, %135) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %136 = "byre.alias"(%56) {offset = 4 : i64} : (memref<31254528xi8>) -> memref<f32>
    byre.compute @ReduceSumOpf32f32(%130, %136) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%135, %136, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown9"} : memref<f32>, memref<f32>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%130, %135) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %137 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<f32>
    byre.compute @PTXOp(%135, %137) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown10"} : memref<f32>, memref<f32>
    byre.compute @PTXOp(%137, %133, %130) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown11"} : memref<f32>, memref<256x30522xf32>, memref<256x30522xf32>
    byre.compute @ReduceSumOpf32f32(%130, %129) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    byre.compute @PTXOp(%129, %134, %130, %127) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown12"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %138 = "byre.alias"(%56) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x30522xf32>
    %139 = "byre.alias"(%55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%126, %127, %139) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    %140 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%127, %arg4, %140) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    %141 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    %142 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%141, %120, %arg43, %124, %125, %142, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%142, %116, %arg41, %121, %122, %141, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %143 = "byre.alias"(%53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%141, %119, %arg39, %117, %118, %142, %arg83, %arg84, %143) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %144 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%142, %113, %arg37, %144, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%144, %109, %arg35, %114, %115, %142, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %145 = "byre.alias"(%52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @PTXOp(%143, %142, %145) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %146 = "byre.alias"(%51) {offset = 0 : i64} : (memref<524288xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%145, %112, %arg33, %110, %111, %141, %arg77, %arg78, %146) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%141, %108, %arg31, %142, %arg75, %arg76) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %147 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x128x2x64xf32>
    %148 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%147, %148) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %149 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    %150 = "byre.alias"(%53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%148, %104, %106, %149, %150) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %151 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%149, %104, %105, %151) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %152 = "byre.alias"(%53) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    %153 = "byre.alias"(%52) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%151, %102, %103, %152, %153) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%152, %98, %arg25, %142, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @ftv4.linear_transpose_backward(%150, %98, %arg29, %141, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %154 = "byre.alias"(%55) {offset = 15758336 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%153, %98, %arg27, %154, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @PTXOp(%146, %142, %141, %154, %143) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%143, %101, %arg23, %99, %100, %142, %arg67, %arg68, %145) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%142, %95, %arg21, %144, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%144, %91, %arg19, %96, %97, %142, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    byre.compute @PTXOp(%145, %142, %141) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%141, %94, %arg17, %92, %93, %142, %arg61, %arg62, %145) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%142, %89, %arg15, %141, %arg59, %arg60) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %155 = "byre.alias"(%55) {offset = 15627264 : i64} : (memref<31254528xi8>) -> memref<2x128x2x64xf32>
    %156 = "byre.alias"(%54) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%155, %156) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%156, %83, %86, %151, %150) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.softmax_backward(%151, %83, %85, %149) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %157 = "byre.alias"(%55) {offset = 15758336 : i64} : (memref<31254528xi8>) -> memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%149, %80, %81, %157, %148) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose_backward(%157, %77, %arg9, %142, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %158 = "byre.alias"(%54) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%150, %77, %arg13, %158, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %159 = "byre.alias"(%54) {offset = 262144 : i64} : (memref<31254528xi8>) -> memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%148, %77, %arg11, %159, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    byre.compute @PTXOp(%145, %142, %158, %159, %141) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%141, %76, %arg7, %78, %79, %142, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %160 = "byre.alias"(%54) {offset = 131072 : i64} : (memref<31254528xi8>) -> memref<256x128xf32>
    byre.compute @PTXOp(%66, %142, %70, %140, %160) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18"} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%139, %65, %140, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%58, %69, %160, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    %161 = "byre.alias"(%53) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @ReduceSumOpf32f32(%142, %161) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %162 = "byre.alias"(%55) {offset = 0 : i64} : (memref<31254528xi8>) -> memref<128x128xf32>
    byre.compute @PTXOp(%74, %161, %162) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%57, %73, %162, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%138, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

