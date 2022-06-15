// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cuda" | FileCheck %s

// CHECK-LABEL: func @main
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
      %5 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
      %6 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
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
        %28 = memref.load %6[%27, %21] : memref<2x128xi1>
        %29 = memref.load %arg1[%27, %21, %11] : memref<2x128x128xf32>
        %30 = arith.select %28, %29, %cst : f32
        memref.store %30, %arg3[%17, %11] : memref<256x128xf32>
        %31 = memref.load %5[%27, %21] : memref<2x128xi1>
        %32 = arith.select %31, %29, %cst : f32
        memref.store %32, %arg4[%17, %11] : memref<256x128xf32>
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
    gpu.func @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>, %arg3: memref<256x30522xf32>, %arg4: memref<2x128x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c128 = arith.constant 128 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.expand_shape %arg0 [[0, 1]] : memref<256xf32> into memref<2x128xf32>
      %6 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %6 {
        %7 = arith.remsi %4, %c30522 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c30522 : index
        %10 = arith.select %8, %9, %7 : index
        %11 = arith.cmpi slt, %4, %c0 : index
        %12 = arith.subi %c-1, %4 : index
        %13 = arith.select %11, %12, %4 : index
        %14 = arith.divsi %13, %c30522 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = memref.load %arg2[%16, %10] : memref<256x30522xf32>
        %18 = memref.load %arg1[%16, %10] : memref<256x30522xf32>
        %19 = memref.load %arg0[%16] : memref<256xf32>
        %20 = arith.mulf %18, %19 : f32
        %21 = arith.subf %17, %20 : f32
        memref.store %21, %arg3[%16, %10] : memref<256x30522xf32>
        %22 = arith.remsi %16, %c128 : index
        %23 = arith.cmpi slt, %22, %c0 : index
        %24 = arith.addi %22, %c128 : index
        %25 = arith.select %23, %24, %22 : index
        %26 = arith.cmpi slt, %16, %c0 : index
        %27 = arith.subi %c-1, %16 : index
        %28 = arith.select %26, %27, %16 : index
        %29 = arith.divsi %28, %c128 : index
        %30 = arith.subi %c-1, %29 : index
        %31 = arith.select %26, %30, %29 : index
        %32 = memref.load %5[%31, %25] : memref<2x128xf32>
        %33 = arith.mulf %18, %32 : f32
        %34 = arith.subf %17, %33 : f32
        memref.store %34, %arg4[%31, %25, %10] : memref<2x128x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown11(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) kernel {
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
    gpu.func @Unknown10(%arg0: memref<f32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) kernel {
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
    gpu.func @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>) kernel {
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
        %16 = memref.load %arg3[%15] : memref<256xi1>
        %17 = memref.load %arg2[%15] : memref<256xi64>
        %18 = arith.index_cast %9 : index to i64
        %19 = arith.cmpi eq, %17, %18 : i64
        %20 = arith.select %19, %cst, %cst_0 : f32
        %21 = arith.select %16, %cst, %cst_0 : f32
        %22 = arith.mulf %21, %20 : f32
        memref.store %22, %arg4[%15, %9] : memref<256x30522xf32>
        %23 = memref.load %arg1[%15, %9] : memref<256x30522xf32>
        %24 = memref.load %arg0[%15] : memref<256xf32>
        %25 = arith.subf %23, %24 : f32
        %26 = arith.negf %20 : f32
        %27 = arith.mulf %26, %25 : f32
        %28 = arith.cmpf une, %20, %cst : f32
        %29 = arith.select %28, %cst_0, %27 : f32
        %30 = arith.mulf %29, %22 : f32
        memref.store %30, %arg5[%15, %9] : memref<256x30522xf32>
        %31 = arith.mulf %26, %22 : f32
        memref.store %31, %arg6[%15, %9] : memref<256x30522xf32>
        %32 = math.exp %25 : f32
        memref.store %32, %arg7[%15, %9] : memref<256x30522xf32>
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
        %19 = math.exp %18 : f32
        memref.store %19, %arg3[%15, %9] : memref<256x30522xf32>
      }
      gpu.return
    }
    gpu.func @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>, %arg2: memref<2x128x30522xf32>, %arg3: memref<256x30522xf32>) kernel {
      %c0 = arith.constant 0 : index
      %c7813632 = arith.constant 7813632 : index
      %c128 = arith.constant 128 : index
      %c30522 = arith.constant 30522 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
      %6 = arith.cmpi slt, %4, %c7813632 : index
      scf.if %6 {
        %7 = arith.remsi %4, %c30522 : index
        %8 = arith.cmpi slt, %7, %c0 : index
        %9 = arith.addi %7, %c30522 : index
        %10 = arith.select %8, %9, %7 : index
        %11 = arith.cmpi slt, %4, %c0 : index
        %12 = arith.subi %c-1, %4 : index
        %13 = arith.select %11, %12, %4 : index
        %14 = arith.divsi %13, %c30522 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = arith.remsi %16, %c128 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c128 : index
        %20 = arith.select %18, %19, %17 : index
        %21 = arith.cmpi slt, %16, %c0 : index
        %22 = arith.subi %c-1, %16 : index
        %23 = arith.select %21, %22, %16 : index
        %24 = arith.divsi %23, %c128 : index
        %25 = arith.subi %c-1, %24 : index
        %26 = arith.select %21, %25, %24 : index
        %27 = memref.load %arg0[%16, %10] : memref<256x30522xf32>
        %28 = memref.load %arg1[%10] : memref<30522xf32>
        %29 = arith.addf %27, %28 : f32
        memref.store %29, %arg2[%26, %20, %10] : memref<2x128x30522xf32>
        %30 = memref.load %5[%26, %20, %10] : memref<2x128x30522xf32>
        %31 = arith.addf %30, %28 : f32
        memref.store %31, %arg3[%16, %10] : memref<256x30522xf32>
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
        %26 = memref.load %arg0[%15, %9] : memref<256x128xf32>
        %27 = memref.load %arg1[%15, %9] : memref<256x128xf32>
        %28 = memref.load %arg2[%19, %9] : memref<128x128xf32>
        %29 = arith.addf %26, %27 : f32
        %30 = arith.addf %29, %28 : f32
        memref.store %30, %arg3[%25, %19, %9] : memref<2x128x128xf32>
      }
      gpu.return
    }
    gpu.func @Unknown3(%arg0: memref<1x128xi64>, %arg1: memref<128xui32>, %arg2: memref<128xi64>, %arg3: memref<128xi1>) kernel {
      %c-1_i64 = arith.constant -1 : i64
      %c512_i64 = arith.constant 512 : i64
      %c0_i64 = arith.constant 0 : i64
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
      %c-1_i64 = arith.constant -1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c0_i64 = arith.constant 0 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%9] : memref<128xi64>
        %17 = arith.trunci %16 : i64 to i32
        %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
        memref.store %18, %arg1[%15, %9] : memref<2x128xui32>
        %19 = arith.addi %16, %c2_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = arith.select %20, %19, %16 : i64
        memref.store %21, %arg2[%15, %9] : memref<2x128xi64>
        %22 = arith.cmpi ne, %16, %c-1_i64 : i64
        memref.store %22, %arg3[%15, %9] : memref<2x128xi1>
      }
      gpu.return
    }
    gpu.func @Unknown1(%arg0: memref<2x128xi64>, %arg1: memref<256xui32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) kernel {
      %c30522_i64 = arith.constant 30522 : i64
      %c0_i64 = arith.constant 0 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.trunci %16 : i64 to i32
        %18 = builtin.unrealized_conversion_cast %17 : i32 to ui32
        memref.store %18, %arg1[%4] : memref<256xui32>
        %19 = arith.addi %16, %c30522_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = arith.select %20, %19, %16 : i64
        memref.store %21, %arg2[%4] : memref<256xi64>
        %22 = arith.cmpi ne, %16, %c0_i64 : i64
        memref.store %22, %arg3[%4] : memref<256xi1>
      }
      gpu.return
    }
    gpu.func @Unknown0(%arg0: memref<2x128xi64>, %arg1: memref<256xi1>) kernel {
      %c-100_i64 = arith.constant -100 : i64
      %c0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c-1 = arith.constant -1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = arith.cmpi slt, %4, %c256 : index
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
        %16 = memref.load %arg0[%15, %9] : memref<2x128xi64>
        %17 = arith.cmpi ne, %16, %c-100_i64 : i64
        memref.store %17, %arg1[%4] : memref<256xi1>
      }
      gpu.return
    }
  }
  func @main(%arg0: memref<2x128xi64> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<2x128xi64> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1x512xi64> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x512xi64> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<30522x128xf32> {byre.argname = "Input4", byre.argtype = 1 : i32}, %arg5: memref<2x128xf32> {byre.argname = "Input5", byre.argtype = 1 : i32}, %arg6: memref<512x128xf32> {byre.argname = "Input6", byre.argtype = 1 : i32}, %arg7: memref<128xf32> {byre.argname = "Input7", byre.argtype = 1 : i32}, %arg8: memref<128xf32> {byre.argname = "Input8", byre.argtype = 1 : i32}, %arg9: memref<128x128xf32> {byre.argname = "Input9", byre.argtype = 1 : i32}, %arg10: memref<128xf32> {byre.argname = "Input10", byre.argtype = 1 : i32}, %arg11: memref<128x128xf32> {byre.argname = "Input11", byre.argtype = 1 : i32}, %arg12: memref<128xf32> {byre.argname = "Input12", byre.argtype = 1 : i32}, %arg13: memref<128x128xf32> {byre.argname = "Input13", byre.argtype = 1 : i32}, %arg14: memref<128xf32> {byre.argname = "Input14", byre.argtype = 1 : i32}, %arg15: memref<128x128xf32> {byre.argname = "Input15", byre.argtype = 1 : i32}, %arg16: memref<128xf32> {byre.argname = "Input16", byre.argtype = 1 : i32}, %arg17: memref<128xf32> {byre.argname = "Input17", byre.argtype = 1 : i32}, %arg18: memref<128xf32> {byre.argname = "Input18", byre.argtype = 1 : i32}, %arg19: memref<512x128xf32> {byre.argname = "Input19", byre.argtype = 1 : i32}, %arg20: memref<512xf32> {byre.argname = "Input20", byre.argtype = 1 : i32}, %arg21: memref<128x512xf32> {byre.argname = "Input21", byre.argtype = 1 : i32}, %arg22: memref<128xf32> {byre.argname = "Input22", byre.argtype = 1 : i32}, %arg23: memref<128xf32> {byre.argname = "Input23", byre.argtype = 1 : i32}, %arg24: memref<128xf32> {byre.argname = "Input24", byre.argtype = 1 : i32}, %arg25: memref<128x128xf32> {byre.argname = "Input25", byre.argtype = 1 : i32}, %arg26: memref<128xf32> {byre.argname = "Input26", byre.argtype = 1 : i32}, %arg27: memref<128x128xf32> {byre.argname = "Input27", byre.argtype = 1 : i32}, %arg28: memref<128xf32> {byre.argname = "Input28", byre.argtype = 1 : i32}, %arg29: memref<128x128xf32> {byre.argname = "Input29", byre.argtype = 1 : i32}, %arg30: memref<128xf32> {byre.argname = "Input30", byre.argtype = 1 : i32}, %arg31: memref<128x128xf32> {byre.argname = "Input31", byre.argtype = 1 : i32}, %arg32: memref<128xf32> {byre.argname = "Input32", byre.argtype = 1 : i32}, %arg33: memref<128xf32> {byre.argname = "Input33", byre.argtype = 1 : i32}, %arg34: memref<128xf32> {byre.argname = "Input34", byre.argtype = 1 : i32}, %arg35: memref<512x128xf32> {byre.argname = "Input35", byre.argtype = 1 : i32}, %arg36: memref<512xf32> {byre.argname = "Input36", byre.argtype = 1 : i32}, %arg37: memref<128x512xf32> {byre.argname = "Input37", byre.argtype = 1 : i32}, %arg38: memref<128xf32> {byre.argname = "Input38", byre.argtype = 1 : i32}, %arg39: memref<128xf32> {byre.argname = "Input39", byre.argtype = 1 : i32}, %arg40: memref<128xf32> {byre.argname = "Input40", byre.argtype = 1 : i32}, %arg41: memref<128x128xf32> {byre.argname = "Input41", byre.argtype = 1 : i32}, %arg42: memref<128xf32> {byre.argname = "Input42", byre.argtype = 1 : i32}, %arg43: memref<128xf32> {byre.argname = "Input43", byre.argtype = 1 : i32}, %arg44: memref<128xf32> {byre.argname = "Input44", byre.argtype = 1 : i32}, %arg45: memref<30522xf32> {byre.argname = "Input45", byre.argtype = 1 : i32}, %arg46: memref<2x128x30522xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg47: memref<f32> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg48: memref<30522x128xf32> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg49: memref<2x128xf32> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg50: memref<512x128xf32> {byre.argname = "Output4", byre.argtype = 2 : i32}, %arg51: memref<128xf32> {byre.argname = "Output5", byre.argtype = 2 : i32}, %arg52: memref<128xf32> {byre.argname = "Output6", byre.argtype = 2 : i32}, %arg53: memref<128x128xf32> {byre.argname = "Output7", byre.argtype = 2 : i32}, %arg54: memref<128xf32> {byre.argname = "Output8", byre.argtype = 2 : i32}, %arg55: memref<128x128xf32> {byre.argname = "Output9", byre.argtype = 2 : i32}, %arg56: memref<128xf32> {byre.argname = "Output10", byre.argtype = 2 : i32}, %arg57: memref<128x128xf32> {byre.argname = "Output11", byre.argtype = 2 : i32}, %arg58: memref<128xf32> {byre.argname = "Output12", byre.argtype = 2 : i32}, %arg59: memref<128x128xf32> {byre.argname = "Output13", byre.argtype = 2 : i32}, %arg60: memref<128xf32> {byre.argname = "Output14", byre.argtype = 2 : i32}, %arg61: memref<128xf32> {byre.argname = "Output15", byre.argtype = 2 : i32}, %arg62: memref<128xf32> {byre.argname = "Output16", byre.argtype = 2 : i32}, %arg63: memref<512x128xf32> {byre.argname = "Output17", byre.argtype = 2 : i32}, %arg64: memref<512xf32> {byre.argname = "Output18", byre.argtype = 2 : i32}, %arg65: memref<128x512xf32> {byre.argname = "Output19", byre.argtype = 2 : i32}, %arg66: memref<128xf32> {byre.argname = "Output20", byre.argtype = 2 : i32}, %arg67: memref<128xf32> {byre.argname = "Output21", byre.argtype = 2 : i32}, %arg68: memref<128xf32> {byre.argname = "Output22", byre.argtype = 2 : i32}, %arg69: memref<128x128xf32> {byre.argname = "Output23", byre.argtype = 2 : i32}, %arg70: memref<128xf32> {byre.argname = "Output24", byre.argtype = 2 : i32}, %arg71: memref<128x128xf32> {byre.argname = "Output25", byre.argtype = 2 : i32}, %arg72: memref<128xf32> {byre.argname = "Output26", byre.argtype = 2 : i32}, %arg73: memref<128x128xf32> {byre.argname = "Output27", byre.argtype = 2 : i32}, %arg74: memref<128xf32> {byre.argname = "Output28", byre.argtype = 2 : i32}, %arg75: memref<128x128xf32> {byre.argname = "Output29", byre.argtype = 2 : i32}, %arg76: memref<128xf32> {byre.argname = "Output30", byre.argtype = 2 : i32}, %arg77: memref<128xf32> {byre.argname = "Output31", byre.argtype = 2 : i32}, %arg78: memref<128xf32> {byre.argname = "Output32", byre.argtype = 2 : i32}, %arg79: memref<512x128xf32> {byre.argname = "Output33", byre.argtype = 2 : i32}, %arg80: memref<512xf32> {byre.argname = "Output34", byre.argtype = 2 : i32}, %arg81: memref<128x512xf32> {byre.argname = "Output35", byre.argtype = 2 : i32}, %arg82: memref<128xf32> {byre.argname = "Output36", byre.argtype = 2 : i32}, %arg83: memref<128xf32> {byre.argname = "Output37", byre.argtype = 2 : i32}, %arg84: memref<128xf32> {byre.argname = "Output38", byre.argtype = 2 : i32}, %arg85: memref<128x128xf32> {byre.argname = "Output39", byre.argtype = 2 : i32}, %arg86: memref<128xf32> {byre.argname = "Output40", byre.argtype = 2 : i32}, %arg87: memref<128xf32> {byre.argname = "Output41", byre.argtype = 2 : i32}, %arg88: memref<128xf32> {byre.argname = "Output42", byre.argtype = 2 : i32}, %arg89: memref<30522xf32> {byre.argname = "Output43", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
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
    %60 = memref.alloc() : memref<128xi64>
    byre.compute @AliasOp(%arg2, %60) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<128xi64>
    %61 = memref.alloc() : memref<1x128xi64>
    byre.compute @AliasOp(%arg3, %61) {arg_alias, offset = 0 : i32} : memref<1x512xi64>, memref<1x128xi64>
    %62 = memref.alloc() : memref<256xi64>
    byre.compute @AliasOp(%arg1, %62) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %63 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%6, %63) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%arg1, %63) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32], kernel_name = "Unknown0"} : memref<2x128xi64>, memref<256xi1>
    byre.compute @AliasOp(%arg1, %62) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<256xi64>
    %64 = memref.alloc() : memref<256xui32>
    byre.compute @AliasOp(%56, %64) {offset = 0 : index} : memref<31254528xi8>, memref<256xui32>
    %65 = memref.alloc() : memref<256x1xi64>
    byre.compute @AliasOp(%21, %65) {offset = 0 : index} : memref<2048xi8>, memref<256x1xi64>
    %66 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%5, %66) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%arg0, %64, %65, %66) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown1"} : memref<2x128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %67 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%55, %67) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg4, %64, %67) {dim = 0 : i32} : memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>
    %68 = memref.alloc() : memref<256xui32>
    byre.compute @AliasOp(%55, %68) {offset = 131072 : index} : memref<31254528xi8>, memref<256xui32>
    %69 = memref.alloc() : memref<256x1xi64>
    byre.compute @AliasOp(%22, %69) {offset = 0 : index} : memref<2048xi8>, memref<256x1xi64>
    %70 = memref.alloc() : memref<256xi1>
    byre.compute @AliasOp(%4, %70) {offset = 0 : index} : memref<32xi8>, memref<256xi1>
    byre.compute @PTXOp(%60, %68, %69, %70) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown2"} : memref<128xi64>, memref<256xui32>, memref<256x1xi64>, memref<256xi1>
    %71 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%56, %71) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg5, %68, %71) {dim = 0 : i32} : memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>
    %72 = memref.alloc() : memref<128xui32>
    byre.compute @AliasOp(%55, %72) {offset = 131072 : index} : memref<31254528xi8>, memref<128xui32>
    %73 = memref.alloc() : memref<128x1xi64>
    byre.compute @AliasOp(%19, %73) {offset = 0 : index} : memref<1024xi8>, memref<128x1xi64>
    %74 = memref.alloc() : memref<128xi1>
    byre.compute @AliasOp(%3, %74) {offset = 0 : index} : memref<16xi8>, memref<128xi1>
    byre.compute @PTXOp(%61, %72, %73, %74) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 1 : i32, 1 : i32, 1 : i32], kernel_name = "Unknown3"} : memref<1x128xi64>, memref<128xui32>, memref<128x1xi64>, memref<128xi1>
    %75 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%56, %75) {offset = 131072 : index} : memref<31254528xi8>, memref<128x128xf32>
    byre.compute @IndexSelectOpf32ui32f32(%arg6, %72, %75) {dim = 0 : i32} : memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>
    %76 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%33, %76) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%67, %71, %75, %76) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown4"} : memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%40, %77) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %78 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%18, %78) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %79 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%17, %79) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @ftv4.layernorm(%76, %arg7, %arg8, %77, %78, %79) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %80 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%35, %80) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg9, %arg10, %80) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %81 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%36, %81) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg11, %arg12, %81) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %82 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%55, %82) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%80, %81, %82) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %83 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%46, %83) {offset = 0 : index} : memref<262144xi8>, memref<2x2x128x128xf32>
    %84 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%56, %84) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %85 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @AliasOp(%23, %85) {offset = 0 : index} : memref<65536xi8>, memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%82, %59, %83, %84, %85) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %86 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%37, %86) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%77, %arg13, %arg14, %86) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %87 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%56, %87) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%83, %86, %87) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %88 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%38, %88) {offset = 0 : index} : memref<131072xi8>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%87, %88) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %89 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%38, %89) {offset = 0 : i32} : memref<131072xi8>, memref<2x128x128xf32>
    %90 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %90) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%89, %arg15, %arg16, %90) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %91 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%39, %91) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %92 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%16, %92) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %93 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%15, %93) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %94 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%25, %94) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%90, %arg17, %arg18, %77, %91, %92, %93, %94) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %95 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%51, %95) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %96 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%50, %96) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %97 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%2, %97) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%91, %arg19, %arg20, %95, %96, %97) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %98 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %98) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%95, %arg21, %arg22, %98) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %99 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%26, %99) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %100 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%14, %100) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %101 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%13, %101) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %102 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%27, %102) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%98, %arg23, %arg24, %91, %99, %100, %101, %102) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %103 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%28, %103) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%99, %arg25, %arg26, %103) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %104 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%32, %104) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%99, %arg27, %arg28, %104) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %105 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%56, %105) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.matmul(%103, %104, %105) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>
    %106 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%47, %106) {offset = 0 : index} : memref<262144xi8>, memref<2x2x128x128xf32>
    %107 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%56, %107) {offset = 262144 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %108 = memref.alloc() : memref<2x2x128x128xui8>
    byre.compute @AliasOp(%24, %108) {offset = 0 : index} : memref<65536xi8>, memref<2x2x128x128xui8>
    byre.compute @ftv4.softmax(%105, %59, %106, %107, %108) {batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32} : memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>
    %109 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%29, %109) {offset = 0 : index} : memref<131072xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.linear_transpose(%99, %arg29, %arg30, %109) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>
    %110 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%56, %110) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul(%106, %109, %110) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %111 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%30, %111) {offset = 0 : index} : memref<131072xi8>, memref<2x128x2x64xf32>
    byre.compute @ftv4.transpose4d(%110, %111) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x2x128x64xf32>, memref<2x128x2x64xf32>
    %112 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%30, %112) {offset = 0 : i32} : memref<131072xi8>, memref<2x128x128xf32>
    %113 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %113) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%112, %arg31, %arg32, %113) : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %114 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%31, %114) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %115 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%12, %115) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %116 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%11, %116) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %117 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%34, %117) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%113, %arg33, %arg34, %99, %114, %115, %116, %117) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %118 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%49, %118) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %119 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%48, %119) {offset = 0 : index} : memref<524288xi8>, memref<2x128x512xf32>
    %120 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%0, %120) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%114, %arg35, %arg36, %118, %119, %120) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>
    %121 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %121) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear(%118, %arg37, %arg38, %121) : memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>
    %122 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%41, %122) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %123 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%10, %123) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %124 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%9, %124) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %125 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%42, %125) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_residual(%121, %arg39, %arg40, %114, %122, %123, %124, %125) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>
    %126 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%43, %126) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %127 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%44, %127) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %128 = memref.alloc() : memref<0xf32>
    byre.compute @AliasOp(%1, %128) {offset = 0 : index} : memref<0xi8>, memref<0xf32>
    byre.compute @ftv4.linear_gelu_dropout(%122, %arg41, %arg42, %126, %127, %128) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>
    %129 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%45, %129) {offset = 0 : index} : memref<131072xi8>, memref<2x128x128xf32>
    %130 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%20, %130) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    %131 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%8, %131) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @ftv4.layernorm(%126, %arg43, %arg44, %129, %130, %131) : memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>
    %132 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%45, %132) {offset = 0 : i32} : memref<131072xi8>, memref<256x128xf32>
    %133 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%56, %133) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @MatmulOpf32f32f32(%132, %arg4, %133) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 1 : i64} : memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>
    %134 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%55, %134) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%133, %arg45, %arg46, %134) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [2 : i32, 1 : i32, 3 : i32, 2 : i32], kernel_name = "Unknown5"} : memref<256x30522xf32>, memref<30522xf32>, memref<2x128x30522xf32>, memref<256x30522xf32>
    %135 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%53, %135) {offset = 0 : index} : memref<31254528xi8>, memref<256xf32>
    byre.compute @ReduceMaxOpf32f32(%134, %135) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %136 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%56, %136) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %137 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %137) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%135, %134, %136, %137) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown6"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %138 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%55, %138) {offset = 0 : index} : memref<31254528xi8>, memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%137, %138) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %139 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%7, %139) {offset = 0 : index} : memref<1024xi8>, memref<256xf32>
    byre.compute @PTXOp(%138, %139) {BlockSize.x = 128 : i32, GridSize.x = 2 : i32, arg_ranks = [1 : i32, 1 : i32], kernel_name = "Unknown7"} : memref<256xf32>, memref<256xf32>
    %140 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%52, %140) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %141 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%53, %141) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %142 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %142) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %143 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%55, %143) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%139, %136, %62, %63, %140, %141, %142, %143) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown8"} : memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
    %144 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%arg1, %144) {arg_alias, offset = 0 : i32} : memref<2x128xi64>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%140, %144) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %145 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%56, %145) {offset = 0 : index} : memref<31254528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%140, %145) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    %146 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%52, %146) {offset = 0 : index} : memref<31254528xi8>, memref<f32>
    byre.compute @PTXOp(%145, %146) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32], kernel_name = "Unknown9"} : memref<f32>, memref<f32>
    %147 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%56, %147) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    byre.compute @PTXOp(%146, %142, %147) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [0 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown10"} : memref<f32>, memref<256x30522xf32>, memref<256x30522xf32>
    %148 = memref.alloc() : memref<256xf32>
    byre.compute @AliasOp(%52, %148) {offset = 0 : index} : memref<31254528xi8>, memref<256xf32>
    byre.compute @ReduceSumOpf32f32(%147, %148) {dimensions = dense<1> : tensor<1xi64>} : memref<256x30522xf32>, memref<256xf32>
    %149 = memref.alloc() : memref<f32>
    byre.compute @AliasOp(%54, %149) {offset = 0 : index} : memref<31254528xi8>, memref<f32>
    byre.compute @ReduceSumOpf32f32(%141, %149) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<256x30522xf32>, memref<f32>
    byre.compute @PTXOp(%149, %144, %arg47) {BlockSize.x = 128 : i32, GridSize.x = 1 : i32, arg_ranks = [0 : i32, 0 : i32, 0 : i32], kernel_name = "Unknown11"} : memref<f32>, memref<f32>, memref<f32>
    %150 = memref.alloc() : memref<256x30522xf32>
    byre.compute @AliasOp(%54, %150) {offset = 0 : index} : memref<31254528xi8>, memref<256x30522xf32>
    %151 = memref.alloc() : memref<2x128x30522xf32>
    byre.compute @AliasOp(%53, %151) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x30522xf32>
    byre.compute @PTXOp(%148, %143, %147, %150, %151) {BlockSize.x = 128 : i32, GridSize.x = 61044 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32, 2 : i32, 3 : i32], kernel_name = "Unknown12"} : memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<2x128x30522xf32>
    %152 = memref.alloc() : memref<30522x128xf32>
    byre.compute @AliasOp(%56, %152) {offset = 0 : index} : memref<31254528xi8>, memref<30522x128xf32>
    byre.compute @MatmulOpf32f32f32(%132, %150, %152) {lhs_contracting_dimension = 0 : i64, output_transpose, rhs_contracting_dimension = 0 : i64} : memref<256x128xf32>, memref<256x30522xf32>, memref<30522x128xf32>
    %153 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%56, %153) {offset = 15627264 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @MatmulOpf32f32f32(%150, %arg4, %153) {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} : memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>
    %154 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %154) {offset = 15627264 : i32} : memref<31254528xi8>, memref<2x128x128xf32>
    %155 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %155) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%154, %126, %arg43, %130, %131, %155, %arg87, %arg88) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %156 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %156) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%155, %122, %arg41, %127, %128, %156, %arg85, %arg86) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %157 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %157) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    %158 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%52, %158) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%156, %125, %arg39, %123, %124, %157, %arg83, %arg84, %158) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %159 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%56, %159) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%157, %118, %arg37, %159, %arg81, %arg82) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %160 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %160) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%159, %114, %arg35, %119, %120, %160, %arg79, %arg80) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %161 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %161) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%158, %160, %161) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown14"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %162 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %162) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    %163 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%52, %163) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%161, %117, %arg33, %115, %116, %162, %arg77, %arg78, %163) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %164 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %164) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%162, %112, %arg31, %164, %arg75, %arg76) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %165 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%55, %165) {offset = 0 : i32} : memref<31254528xi8>, memref<2x128x2x64xf32>
    %166 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%56, %166) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%165, %166) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %167 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%55, %167) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %168 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %168) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%166, %106, %109, %167, %168) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %169 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%56, %169) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%167, %106, %108, %169) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %170 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%55, %170) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    %171 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%52, %171) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%169, %103, %104, %170, %171) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %172 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%49, %172) {offset = 0 : index} : memref<524288xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%170, %99, %arg25, %172, %arg69, %arg70) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %173 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %173) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%168, %99, %arg29, %173, %arg73, %arg74) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %174 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %174) {offset = 131072 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%171, %99, %arg27, %174, %arg71, %arg72) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %175 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %175) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%163, %172, %173, %174, %175) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown15"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %176 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %176) {offset = 131072 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    %177 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %177) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%175, %102, %arg23, %100, %101, %176, %arg67, %arg68, %177) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %178 = memref.alloc() : memref<2x128x512xf32>
    byre.compute @AliasOp(%56, %178) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x512xf32>
    byre.compute @ftv4.linear_backward(%176, %95, %arg21, %178, %arg65, %arg66) : memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>
    %179 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %179) {offset = 131072 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_gelu_dropout_backward(%178, %91, %arg19, %96, %97, %179, %arg63, %arg64) {act_gelu = true, dropout_rate = 0.000000e+00 : f32} : memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>
    %180 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %180) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%177, %179, %180) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown16"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %181 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %181) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    %182 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %182) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward_residual(%180, %94, %arg17, %92, %93, %181, %arg61, %arg62, %182) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>
    %183 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %183) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_backward(%181, %89, %arg15, %183, %arg59, %arg60) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %184 = memref.alloc() : memref<2x128x2x64xf32>
    byre.compute @AliasOp(%56, %184) {offset = 15627264 : i32} : memref<31254528xi8>, memref<2x128x2x64xf32>
    %185 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%54, %185) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.transpose4d_backward(%184, %185) {forward_transpose_type = "TRANSPOSE0213"} : memref<2x128x2x64xf32>, memref<2x2x128x64xf32>
    %186 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%55, %186) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    %187 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%52, %187) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%185, %83, %86, %186, %187) {scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false} : memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>
    %188 = memref.alloc() : memref<2x2x128x128xf32>
    byre.compute @AliasOp(%56, %188) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x2x128x128xf32>
    byre.compute @ftv4.softmax_backward(%186, %83, %85, %188) {dropout_rate = 0.000000e+00 : f32} : memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>
    %189 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%55, %189) {offset = 131072 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    %190 = memref.alloc() : memref<2x2x128x64xf32>
    byre.compute @AliasOp(%55, %190) {offset = 0 : index} : memref<31254528xi8>, memref<2x2x128x64xf32>
    byre.compute @ftv4.matmul_backward(%188, %80, %81, %189, %190) {scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true} : memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>
    %191 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %191) {offset = 15627264 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%189, %77, %arg9, %191, %arg53, %arg54) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %192 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %192) {offset = 15758336 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%187, %77, %arg13, %192, %arg57, %arg58) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %193 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%56, %193) {offset = 15889408 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.linear_transpose_backward(%190, %77, %arg11, %193, %arg55, %arg56) {forward_transpose_type = "TRANSPOSE0213", head_num = 2 : i32} : memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>
    %194 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%55, %194) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @PTXOp(%182, %191, %192, %193, %194) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown17"} : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>
    %195 = memref.alloc() : memref<2x128x128xf32>
    byre.compute @AliasOp(%54, %195) {offset = 0 : index} : memref<31254528xi8>, memref<2x128x128xf32>
    byre.compute @ftv4.layernorm_backward(%194, %76, %arg7, %78, %79, %195, %arg51, %arg52) : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>
    %196 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%56, %196) {offset = 15627264 : index} : memref<31254528xi8>, memref<256x128xf32>
    %197 = memref.alloc() : memref<256x128xf32>
    byre.compute @AliasOp(%55, %197) {offset = 0 : index} : memref<31254528xi8>, memref<256x128xf32>
    byre.compute @PTXOp(%66, %195, %70, %196, %197) {BlockSize.x = 128 : i32, GridSize.x = 256 : i32, arg_ranks = [1 : i32, 3 : i32, 1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown18"} : memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>, memref<256x128xf32>, memref<256x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%152, %65, %196, %arg48) {dim = 0 : i32} : memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%58, %69, %197, %arg49) {dim = 0 : i32} : memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>
    %198 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%55, %198) {offset = 0 : index} : memref<31254528xi8>, memref<128x128xf32>
    byre.compute @ReduceSumOpf32f32(%195, %198) {dimensions = dense<0> : tensor<1xi64>} : memref<2x128x128xf32>, memref<128x128xf32>
    %199 = memref.alloc() : memref<128x128xf32>
    byre.compute @AliasOp(%56, %199) {offset = 0 : index} : memref<31254528xi8>, memref<128x128xf32>
    byre.compute @PTXOp(%74, %198, %199) {BlockSize.x = 128 : i32, GridSize.x = 128 : i32, arg_ranks = [1 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown19"} : memref<128xi1>, memref<128x128xf32>, memref<128x128xf32>
    byre.compute @IndexPutOpf32i64f32f32(%57, %73, %199, %arg50) {dim = 0 : i32} : memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>
    byre.compute @ReduceSumOpf32f32(%151, %arg89) {dimensions = dense<[0, 1]> : tensor<2xi64>} : memref<2x128x30522xf32>, memref<30522xf32>
    return
  }
}

