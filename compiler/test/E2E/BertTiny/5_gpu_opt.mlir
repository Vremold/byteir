// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func.func @main
module {
  func.func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-100_i64 = arith.constant -100 : i64
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %arg0[%5, %9] : memref<2x128xi64>
      %11 = arith.cmpi ne, %10, %c-100_i64 : i64
      memref.store %11, %alloc[%arg1] : memref<256xi1>
    }
    return %collapse_shape, %alloc : memref<256xi64>, memref<256xi1>
  }
  func.func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c30522_i64 = arith.constant 30522 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256xui32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %arg0[%5, %9] : memref<2x128xi64>
      %11 = arith.trunci %10 : i64 to i32
      %12 = builtin.unrealized_conversion_cast %11 : i32 to ui32
      memref.store %12, %alloc[%arg1] : memref<256xui32>
    }
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<256x1xi64>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %arg0[%5, %9] : memref<2x128xi64>
      %11 = arith.addi %10, %c30522_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.select %12, %11, %10 : i64
      memref.store %13, %alloc_0[%arg1, %c0] : memref<256x1xi64>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %arg0[%5, %9] : memref<2x128xi64>
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      memref.store %11, %alloc_1[%arg1] : memref<256xi1>
    }
    return %alloc, %alloc_0, %alloc_1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128xi64>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.remsi %arg1, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg1, %c0 : index
      %5 = arith.subi %c-1, %arg1 : index
      %6 = arith.select %4, %5, %arg1 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%3] : memref<128xi64>
      memref.store %10, %alloc[%9, %3] : memref<2x128xi64>
    }
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<256xui32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %alloc[%5, %9] : memref<2x128xi64>
      %11 = arith.trunci %10 : i64 to i32
      %12 = builtin.unrealized_conversion_cast %11 : i32 to ui32
      memref.store %12, %alloc_0[%arg1] : memref<256xui32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256x1xi64>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %alloc[%5, %9] : memref<2x128xi64>
      %11 = arith.addi %10, %c2_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.select %12, %11, %10 : i64
      memref.store %13, %alloc_1[%arg1, %c0] : memref<256x1xi64>
    }
    %alloc_2 = memref.alloc() {alignment = 128 : i64} : memref<256xi1>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg1, %c128 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c128 : index
      %9 = arith.select %7, %8, %6 : index
      %10 = memref.load %alloc[%5, %9] : memref<2x128xi64>
      %11 = arith.cmpi ne, %10, %c-1_i64 : i64
      memref.store %11, %alloc_2[%arg1] : memref<256xi1>
    }
    return %alloc_0, %alloc_1, %alloc_2 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<128xui32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.addi %arg1, %c128 : index
      %7 = arith.select %0, %6, %arg1 : index
      %8 = memref.load %arg0[%5, %7] : memref<1x128xi64>
      %9 = arith.trunci %8 : i64 to i32
      %10 = builtin.unrealized_conversion_cast %9 : i32 to ui32
      memref.store %10, %alloc[%arg1] : memref<128xui32>
    }
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<128x1xi64>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.addi %arg1, %c128 : index
      %7 = arith.select %0, %6, %arg1 : index
      %8 = memref.load %arg0[%5, %7] : memref<1x128xi64>
      %9 = arith.addi %8, %c512_i64 : i64
      %10 = arith.cmpi slt, %8, %c0_i64 : i64
      %11 = arith.select %10, %9, %8 : i64
      memref.store %11, %alloc_0[%arg1, %c0] : memref<128x1xi64>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128xi1>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %0 = arith.cmpi slt, %arg1, %c0 : index
      %1 = arith.subi %c-1, %arg1 : index
      %2 = arith.select %0, %1, %arg1 : index
      %3 = arith.divsi %2, %c128 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.addi %arg1, %c128 : index
      %7 = arith.select %0, %6, %arg1 : index
      %8 = memref.load %arg0[%5, %7] : memref<1x128xi64>
      %9 = arith.cmpi ne, %8, %c-1_i64 : i64
      memref.store %9, %alloc_1[%arg1] : memref<128xi1>
    }
    return %alloc, %alloc_0, %alloc_1 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func.func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    scf.for %arg3 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg3, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.muli %19, %c128 : index
      %21 = arith.addi %20, %13 : index
      %22 = memref.load %arg0[%21, %3] : memref<256x128xf32>
      %23 = memref.load %arg1[%21, %3] : memref<256x128xf32>
      %24 = memref.load %arg2[%13, %3] : memref<128x128xf32>
      %25 = arith.addf %22, %23 : f32
      %26 = arith.addf %25, %24 : f32
      memref.store %26, %alloc[%19, %13, %3] : memref<2x128x128xf32>
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c7813632 = arith.constant 7813632 : index
    %c1 = arith.constant 1 : index
    %c30522 = arith.constant 30522 : index
    %c-1 = arith.constant -1 : index
    %c128 = arith.constant 128 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x30522xf32>
    scf.for %arg2 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg2, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = arith.muli %19, %c128 : index
      %21 = arith.addi %20, %13 : index
      %22 = memref.load %arg0[%21, %3] : memref<256x30522xf32>
      %23 = memref.load %arg1[%3] : memref<30522xf32>
      %24 = arith.addf %22, %23 : f32
      memref.store %24, %alloc[%19, %13, %3] : memref<2x128x30522xf32>
    }
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<2x128x30522xf32> into memref<256x30522xf32>
    return %alloc, %collapse_shape : memref<2x128x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c7813632 = arith.constant 7813632 : index
    %c1 = arith.constant 1 : index
    %c30522 = arith.constant 30522 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg2 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg2, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg1[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %arg0[%9] : memref<256xf32>
      %12 = arith.subf %10, %11 : f32
      memref.store %12, %alloc[%9, %3] : memref<256x30522xf32>
    }
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg2 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg2, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %alloc[%9, %3] : memref<256x30522xf32>
      %11 = math.exp %10 : f32
      memref.store %11, %alloc_0[%9, %3] : memref<256x30522xf32>
    }
    return %alloc, %alloc_0 : memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<256xf32>
      %1 = math.log %0 : f32
      memref.store %1, %alloc[%arg1] : memref<256xf32>
    }
    return %alloc : memref<256xf32>
  }
  func.func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c7813632 = arith.constant 7813632 : index
    %c1 = arith.constant 1 : index
    %c30522 = arith.constant 30522 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg1[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %arg0[%9] : memref<256xf32>
      %12 = arith.subf %10, %11 : f32
      memref.store %12, %alloc[%9, %3] : memref<256x30522xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg2[%9] : memref<256xi64>
      %11 = arith.index_cast %3 : index to i64
      %12 = arith.cmpi eq, %10, %11 : i64
      %13 = arith.select %12, %cst, %cst_0 : f32
      memref.store %13, %alloc_1[%9, %3] : memref<256x30522xf32>
    }
    %alloc_2 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %alloc_1[%9, %3] : memref<256x30522xf32>
      %11 = arith.negf %10  : f32
      memref.store %11, %alloc_2[%9, %3] : memref<256x30522xf32>
    }
    %alloc_3 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg3[%9] : memref<256xi1>
      %11 = memref.load %alloc_1[%9, %3] : memref<256x30522xf32>
      %12 = arith.select %10, %cst, %cst_0 : f32
      %13 = arith.mulf %12, %11 : f32
      memref.store %13, %alloc_3[%9, %3] : memref<256x30522xf32>
    }
    %alloc_4 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %alloc_1[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %alloc_2[%9, %3] : memref<256x30522xf32>
      %12 = memref.load %alloc[%9, %3] : memref<256x30522xf32>
      %13 = memref.load %alloc_3[%9, %3] : memref<256x30522xf32>
      %14 = arith.mulf %11, %12 : f32
      %15 = arith.cmpf une, %10, %cst : f32
      %16 = arith.select %15, %cst_0, %14 : f32
      %17 = arith.mulf %16, %13 : f32
      memref.store %17, %alloc_4[%9, %3] : memref<256x30522xf32>
    }
    %alloc_5 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %alloc_2[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %alloc_3[%9, %3] : memref<256x30522xf32>
      %12 = arith.mulf %10, %11 : f32
      memref.store %12, %alloc_5[%9, %3] : memref<256x30522xf32>
    }
    %alloc_6 = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg4 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg4, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %alloc[%9, %3] : memref<256x30522xf32>
      %11 = math.exp %10 : f32
      memref.store %11, %alloc_6[%9, %3] : memref<256x30522xf32>
    }
    return %alloc_3, %alloc_4, %alloc_5, %alloc_6 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<f32>
    %0 = memref.load %arg0[] : memref<f32>
    %1 = memref.load %arg1[] : memref<f32>
    %2 = arith.divf %0, %1 : f32
    memref.store %2, %alloc[] : memref<f32>
    return %alloc : memref<f32>
  }
  func.func private @Unknown10(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<f32>
    %0 = memref.load %arg0[] : memref<f32>
    %1 = arith.cmpf une, %0, %cst_0 : f32
    %2 = arith.select %1, %0, %cst : f32
    memref.store %2, %alloc[] : memref<f32>
    return %alloc : memref<f32>
  }
  func.func private @Unknown11(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c7813632 = arith.constant 7813632 : index
    %c1 = arith.constant 1 : index
    %c30522 = arith.constant 30522 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg2 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg2, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg1[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %arg0[] : memref<f32>
      %12 = arith.divf %10, %11 : f32
      memref.store %12, %alloc[%9, %3] : memref<256x30522xf32>
    }
    return %alloc : memref<256x30522xf32>
  }
  func.func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c7813632 = arith.constant 7813632 : index
    %c1 = arith.constant 1 : index
    %c30522 = arith.constant 30522 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x30522xf32>
    scf.for %arg3 = %c0 to %c7813632 step %c1 {
      %0 = arith.remsi %arg3, %c30522 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c30522 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c30522 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg2[%9, %3] : memref<256x30522xf32>
      %11 = memref.load %arg1[%9, %3] : memref<256x30522xf32>
      %12 = memref.load %arg0[%9] : memref<256xf32>
      %13 = arith.mulf %11, %12 : f32
      %14 = arith.subf %10, %13 : f32
      memref.store %14, %alloc[%9, %3] : memref<256x30522xf32>
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    return %alloc, %expand_shape : memref<256x30522xf32>, memref<2x128x30522xf32>
  }
  func.func private @MatmulOp13(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %alloc = memref.alloc() : memref<128x30522xf32>
    "lmhlo.dot"(%arg0, %arg1, %alloc) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %alloc_0 : memref<30522x128xf32>
  }
  func.func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    scf.for %arg2 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg2, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = memref.load %arg0[%19, %13, %3] : memref<2x128x128xf32>
      %21 = memref.load %arg1[%19, %13, %3] : memref<2x128x128xf32>
      %22 = arith.addf %20, %21 : f32
      memref.store %22, %alloc[%19, %13, %3] : memref<2x128x128xf32>
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    scf.for %arg4 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg4, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = memref.load %arg0[%19, %13, %3] : memref<2x128x128xf32>
      %21 = memref.load %arg1[%19, %13, %3] : memref<2x128x128xf32>
      %22 = memref.load %arg2[%19, %13, %3] : memref<2x128x128xf32>
      %23 = memref.load %arg3[%19, %13, %3] : memref<2x128x128xf32>
      %24 = arith.addf %20, %21 : f32
      %25 = arith.addf %24, %22 : f32
      %26 = arith.addf %25, %23 : f32
      memref.store %26, %alloc[%19, %13, %3] : memref<2x128x128xf32>
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    scf.for %arg2 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg2, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = memref.load %arg0[%19, %13, %3] : memref<2x128x128xf32>
      %21 = memref.load %arg1[%19, %13, %3] : memref<2x128x128xf32>
      %22 = arith.addf %20, %21 : f32
      memref.store %22, %alloc[%19, %13, %3] : memref<2x128x128xf32>
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<2x128x128xf32>
    scf.for %arg4 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg4, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg4, %c0 : index
      %5 = arith.subi %c-1, %arg4 : index
      %6 = arith.select %4, %5, %arg4 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = arith.remsi %9, %c128 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c128 : index
      %13 = arith.select %11, %12, %10 : index
      %14 = arith.cmpi slt, %9, %c0 : index
      %15 = arith.subi %c-1, %9 : index
      %16 = arith.select %14, %15, %9 : index
      %17 = arith.divsi %16, %c128 : index
      %18 = arith.subi %c-1, %17 : index
      %19 = arith.select %14, %18, %17 : index
      %20 = memref.load %arg0[%19, %13, %3] : memref<2x128x128xf32>
      %21 = memref.load %arg1[%19, %13, %3] : memref<2x128x128xf32>
      %22 = memref.load %arg2[%19, %13, %3] : memref<2x128x128xf32>
      %23 = memref.load %arg3[%19, %13, %3] : memref<2x128x128xf32>
      %24 = arith.addf %20, %21 : f32
      %25 = arith.addf %24, %22 : f32
      %26 = arith.addf %25, %23 : f32
      memref.store %26, %alloc[%19, %13, %3] : memref<2x128x128xf32>
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    scf.for %arg3 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg3, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9] : memref<256xi1>
      %11 = arith.cmpi slt, %9, %c0 : index
      %12 = arith.subi %c-1, %9 : index
      %13 = arith.select %11, %12, %9 : index
      %14 = arith.divsi %13, %c128 : index
      %15 = arith.subi %c-1, %14 : index
      %16 = arith.select %11, %15, %14 : index
      %17 = arith.remsi %9, %c128 : index
      %18 = arith.cmpi slt, %17, %c0 : index
      %19 = arith.addi %17, %c128 : index
      %20 = arith.select %18, %19, %17 : index
      %21 = memref.load %arg1[%16, %20, %3] : memref<2x128x128xf32>
      %22 = arith.select %10, %21, %cst : f32
      memref.store %22, %alloc[%9, %3] : memref<256x128xf32>
    }
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32>
    scf.for %arg3 = %c0 to %c32768 step %c1 {
      %0 = arith.remsi %arg3, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg3, %c0 : index
      %5 = arith.subi %c-1, %arg3 : index
      %6 = arith.select %4, %5, %arg3 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg2[%9] : memref<256xi1>
      %11 = arith.cmpi slt, %9, %c0 : index
      %12 = arith.subi %c-1, %9 : index
      %13 = arith.select %11, %12, %9 : index
      %14 = arith.divsi %13, %c128 : index
      %15 = arith.subi %c-1, %14 : index
      %16 = arith.select %11, %15, %14 : index
      %17 = arith.remsi %9, %c128 : index
      %18 = arith.cmpi slt, %17, %c0 : index
      %19 = arith.addi %17, %c128 : index
      %20 = arith.select %18, %19, %17 : index
      %21 = memref.load %arg1[%16, %20, %3] : memref<2x128x128xf32>
      %22 = arith.select %10, %21, %cst : f32
      memref.store %22, %alloc_0[%9, %3] : memref<256x128xf32>
    }
    return %alloc, %alloc_0 : memref<256x128xf32>, memref<256x128xf32>
  }
  func.func private @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c16384 = arith.constant 16384 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c-1 = arith.constant -1 : index
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32>
    scf.for %arg2 = %c0 to %c16384 step %c1 {
      %0 = arith.remsi %arg2, %c128 : index
      %1 = arith.cmpi slt, %0, %c0 : index
      %2 = arith.addi %0, %c128 : index
      %3 = arith.select %1, %2, %0 : index
      %4 = arith.cmpi slt, %arg2, %c0 : index
      %5 = arith.subi %c-1, %arg2 : index
      %6 = arith.select %4, %5, %arg2 : index
      %7 = arith.divsi %6, %c128 : index
      %8 = arith.subi %c-1, %7 : index
      %9 = arith.select %4, %8, %7 : index
      %10 = memref.load %arg0[%9] : memref<128xi1>
      %11 = memref.load %arg1[%9, %3] : memref<128x128xf32>
      %12 = arith.select %10, %11, %cst : f32
      memref.store %12, %alloc[%9, %3] : memref<128x128xf32>
    }
    return %alloc : memref<128x128xf32>
  }
  func.func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %alloc = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%alloc_0) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.constant"(%alloc_1) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc_2) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_3 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc_3) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_4 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %alloc_4) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %alloc_5 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%alloc_4, %alloc_5) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %alloc_6 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg3, %alloc_6) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %0:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %1:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %alloc_7 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %1#0, %alloc_7) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %2:3 = call @Unknown2(%alloc_5) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %alloc_8 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg5, %2#0, %alloc_8) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %3:3 = call @Unknown3(%alloc_6) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %alloc_9 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg6, %3#0, %alloc_9) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %4 = call @Unknown4(%alloc_7, %alloc_8, %alloc_9) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    %alloc_10 = memref.alloc() : memref<2x128x128xf32>
    %alloc_11 = memref.alloc() : memref<256xf32>
    %alloc_12 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%4, %arg7, %arg8, %alloc_10, %alloc_11, %alloc_12) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_13 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg9, %arg10, %alloc_13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_14 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg11, %arg12, %alloc_14) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_15 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_13, %alloc_14, %alloc_15) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %alloc_16 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_17 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_18 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%alloc_15, %alloc_1, %alloc_16, %alloc_17, %alloc_18) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = array<i32: 2, 3>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %alloc_19 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg13, %arg14, %alloc_19) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_20 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_16, %alloc_19, %alloc_20) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_21 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%alloc_20, %alloc_21) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_22 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_21, %alloc_22) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %alloc_23 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_22, %arg15, %arg16, %alloc_23) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_24 = memref.alloc() : memref<2x128x128xf32>
    %alloc_25 = memref.alloc() : memref<256xf32>
    %alloc_26 = memref.alloc() : memref<256xf32>
    %alloc_27 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_23, %arg17, %arg18, %alloc_10, %alloc_24, %alloc_25, %alloc_26, %alloc_27) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_28 = memref.alloc() : memref<2x128x512xf32>
    %alloc_29 = memref.alloc() : memref<2x128x512xf32>
    %alloc_30 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_24, %arg19, %arg20, %alloc_28, %alloc_29, %alloc_30) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %alloc_31 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_28, %arg21, %arg22, %alloc_31) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_32 = memref.alloc() : memref<2x128x128xf32>
    %alloc_33 = memref.alloc() : memref<256xf32>
    %alloc_34 = memref.alloc() : memref<256xf32>
    %alloc_35 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_31, %arg23, %arg24, %alloc_24, %alloc_32, %alloc_33, %alloc_34, %alloc_35) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_36 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg25, %arg26, %alloc_36) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_37 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg27, %arg28, %alloc_37) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_38 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_36, %alloc_37, %alloc_38) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %alloc_39 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_40 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_41 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%alloc_38, %alloc_1, %alloc_39, %alloc_40, %alloc_41) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = array<i32: 2, 3>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %alloc_42 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg29, %arg30, %alloc_42) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_43 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_39, %alloc_42, %alloc_43) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_44 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%alloc_43, %alloc_44) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_45 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_44, %alloc_45) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %alloc_46 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_45, %arg31, %arg32, %alloc_46) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_47 = memref.alloc() : memref<2x128x128xf32>
    %alloc_48 = memref.alloc() : memref<256xf32>
    %alloc_49 = memref.alloc() : memref<256xf32>
    %alloc_50 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_46, %arg33, %arg34, %alloc_32, %alloc_47, %alloc_48, %alloc_49, %alloc_50) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_51 = memref.alloc() : memref<2x128x512xf32>
    %alloc_52 = memref.alloc() : memref<2x128x512xf32>
    %alloc_53 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_47, %arg35, %arg36, %alloc_51, %alloc_52, %alloc_53) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %alloc_54 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_51, %arg37, %arg38, %alloc_54) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_55 = memref.alloc() : memref<2x128x128xf32>
    %alloc_56 = memref.alloc() : memref<256xf32>
    %alloc_57 = memref.alloc() : memref<256xf32>
    %alloc_58 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_54, %arg39, %arg40, %alloc_47, %alloc_55, %alloc_56, %alloc_57, %alloc_58) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_59 = memref.alloc() : memref<2x128x128xf32>
    %alloc_60 = memref.alloc() : memref<2x128x128xf32>
    %alloc_61 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_55, %arg41, %arg42, %alloc_59, %alloc_60, %alloc_61) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %alloc_62 = memref.alloc() : memref<2x128x128xf32>
    %alloc_63 = memref.alloc() : memref<256xf32>
    %alloc_64 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%alloc_59, %arg43, %arg44, %alloc_62, %alloc_63, %alloc_64) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_65 = memref.alloc() : memref<256x128xf32>
    "lmhlo.reshape"(%alloc_62, %alloc_65) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    %alloc_66 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.dot"(%alloc_65, %arg4, %alloc_66) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %5:2 = call @Unknown5(%alloc_66, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    %alloc_67 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%5#1, %alloc_2, %alloc_67) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %6:2 = call @Unknown6(%alloc_67, %5#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    %alloc_68 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%6#1, %alloc_3, %alloc_68) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %7 = call @Unknown7(%alloc_68) : (memref<256xf32>) -> memref<256xf32>
    %8:4 = call @Unknown8(%7, %6#0, %0#0, %0#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    %alloc_69 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%8#1, %alloc_3, %alloc_69) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %alloc_70 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%8#0, %alloc_3, %alloc_70) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %9 = call @Unknown9(%alloc_69, %alloc_70) : (memref<f32>, memref<f32>) -> memref<f32>
    %alloc_71 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%8#0, %alloc_3, %alloc_71) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %10 = call @Unknown10(%alloc_71) : (memref<f32>) -> memref<f32>
    %11 = call @Unknown11(%10, %8#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    %alloc_72 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%11, %alloc_3, %alloc_72) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %12:2 = call @Unknown12(%alloc_72, %8#3, %11) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %13 = call @MatmulOp13(%alloc_65, %12#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    %alloc_73 = memref.alloc() : memref<256x128xf32>
    "lmhlo.dot"(%12#0, %arg4, %alloc_73) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    %alloc_74 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_73, %alloc_74) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_75 = memref.alloc() : memref<2x128x128xf32>
    %alloc_76 = memref.alloc() : memref<128xf32>
    %alloc_77 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_74, %alloc_59, %arg43, %alloc_63, %alloc_64, %alloc_75, %alloc_76, %alloc_77) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_78 = memref.alloc() : memref<2x128x128xf32>
    %alloc_79 = memref.alloc() : memref<128x128xf32>
    %alloc_80 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_75, %alloc_55, %arg41, %alloc_60, %alloc_61, %alloc_78, %alloc_79, %alloc_80) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_81 = memref.alloc() : memref<2x128x128xf32>
    %alloc_82 = memref.alloc() : memref<128xf32>
    %alloc_83 = memref.alloc() : memref<128xf32>
    %alloc_84 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_78, %alloc_58, %arg39, %alloc_56, %alloc_57, %alloc_81, %alloc_82, %alloc_83, %alloc_84) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_85 = memref.alloc() : memref<2x128x512xf32>
    %alloc_86 = memref.alloc() : memref<128x512xf32>
    %alloc_87 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_81, %alloc_51, %arg37, %alloc_85, %alloc_86, %alloc_87) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %alloc_88 = memref.alloc() : memref<2x128x128xf32>
    %alloc_89 = memref.alloc() : memref<512x128xf32>
    %alloc_90 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%alloc_85, %alloc_47, %arg35, %alloc_52, %alloc_53, %alloc_88, %alloc_89, %alloc_90) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %14 = call @Unknown14(%alloc_84, %alloc_88) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_91 = memref.alloc() : memref<2x128x128xf32>
    %alloc_92 = memref.alloc() : memref<128xf32>
    %alloc_93 = memref.alloc() : memref<128xf32>
    %alloc_94 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%14, %alloc_50, %arg33, %alloc_48, %alloc_49, %alloc_91, %alloc_92, %alloc_93, %alloc_94) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_95 = memref.alloc() : memref<2x128x128xf32>
    %alloc_96 = memref.alloc() : memref<128x128xf32>
    %alloc_97 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_91, %alloc_45, %arg31, %alloc_95, %alloc_96, %alloc_97) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_98 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%alloc_95, %alloc_98) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_99 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_98, %alloc_99) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_100 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_101 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_99, %alloc_39, %alloc_42, %alloc_100, %alloc_101) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_102 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_100, %alloc_39, %alloc_41, %alloc_102) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %alloc_103 = memref.alloc() : memref<2x2x128x64xf32>
    %alloc_104 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_102, %alloc_36, %alloc_37, %alloc_103, %alloc_104) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_105 = memref.alloc() : memref<2x128x128xf32>
    %alloc_106 = memref.alloc() : memref<128x128xf32>
    %alloc_107 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_103, %alloc_32, %arg25, %alloc_105, %alloc_106, %alloc_107) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_108 = memref.alloc() : memref<2x128x128xf32>
    %alloc_109 = memref.alloc() : memref<128x128xf32>
    %alloc_110 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_101, %alloc_32, %arg29, %alloc_108, %alloc_109, %alloc_110) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_111 = memref.alloc() : memref<2x128x128xf32>
    %alloc_112 = memref.alloc() : memref<128x128xf32>
    %alloc_113 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_104, %alloc_32, %arg27, %alloc_111, %alloc_112, %alloc_113) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %15 = call @Unknown15(%alloc_94, %alloc_105, %alloc_108, %alloc_111) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_114 = memref.alloc() : memref<2x128x128xf32>
    %alloc_115 = memref.alloc() : memref<128xf32>
    %alloc_116 = memref.alloc() : memref<128xf32>
    %alloc_117 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%15, %alloc_35, %arg23, %alloc_33, %alloc_34, %alloc_114, %alloc_115, %alloc_116, %alloc_117) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_118 = memref.alloc() : memref<2x128x512xf32>
    %alloc_119 = memref.alloc() : memref<128x512xf32>
    %alloc_120 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_114, %alloc_28, %arg21, %alloc_118, %alloc_119, %alloc_120) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %alloc_121 = memref.alloc() : memref<2x128x128xf32>
    %alloc_122 = memref.alloc() : memref<512x128xf32>
    %alloc_123 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%alloc_118, %alloc_24, %arg19, %alloc_29, %alloc_30, %alloc_121, %alloc_122, %alloc_123) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %16 = call @Unknown16(%alloc_117, %alloc_121) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_124 = memref.alloc() : memref<2x128x128xf32>
    %alloc_125 = memref.alloc() : memref<128xf32>
    %alloc_126 = memref.alloc() : memref<128xf32>
    %alloc_127 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%16, %alloc_27, %arg17, %alloc_25, %alloc_26, %alloc_124, %alloc_125, %alloc_126, %alloc_127) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_128 = memref.alloc() : memref<2x128x128xf32>
    %alloc_129 = memref.alloc() : memref<128x128xf32>
    %alloc_130 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_124, %alloc_22, %arg15, %alloc_128, %alloc_129, %alloc_130) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.linear_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_131 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%alloc_128, %alloc_131) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_132 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_131, %alloc_132) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = array<i32: 1, 1>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_133 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_134 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_132, %alloc_16, %alloc_19, %alloc_133, %alloc_134) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_135 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_133, %alloc_16, %alloc_18, %alloc_135) {api_version = 1 : i32, backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %alloc_136 = memref.alloc() : memref<2x2x128x64xf32>
    %alloc_137 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_135, %alloc_13, %alloc_14, %alloc_136, %alloc_137) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_138 = memref.alloc() : memref<2x128x128xf32>
    %alloc_139 = memref.alloc() : memref<128x128xf32>
    %alloc_140 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_136, %alloc_10, %arg9, %alloc_138, %alloc_139, %alloc_140) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_141 = memref.alloc() : memref<2x128x128xf32>
    %alloc_142 = memref.alloc() : memref<128x128xf32>
    %alloc_143 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_134, %alloc_10, %arg13, %alloc_141, %alloc_142, %alloc_143) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_144 = memref.alloc() : memref<2x128x128xf32>
    %alloc_145 = memref.alloc() : memref<128x128xf32>
    %alloc_146 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_137, %alloc_10, %arg11, %alloc_144, %alloc_145, %alloc_146) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %17 = call @Unknown17(%alloc_127, %alloc_138, %alloc_141, %alloc_144) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_147 = memref.alloc() : memref<2x128x128xf32>
    %alloc_148 = memref.alloc() : memref<128xf32>
    %alloc_149 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%17, %4, %arg7, %alloc_11, %alloc_12, %alloc_147, %alloc_148, %alloc_149) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %18:2 = call @Unknown18(%1#2, %alloc_147, %2#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %alloc_150 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%13, %1#1, %18#0, %alloc_150) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %alloc_151 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%alloc_0, %2#1, %18#1, %alloc_151) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %alloc_152 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%alloc_147, %alloc_3, %alloc_152) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %19 = call @Unknown19(%3#2, %alloc_152) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    %alloc_153 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%alloc, %3#1, %19, %alloc_153) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    %alloc_154 = memref.alloc() : memref<30522xf32>
    "lmhlo.reduce"(%12#1, %alloc_3, %alloc_154) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %5#0, %9, %alloc_150, %alloc_151, %alloc_153, %alloc_148, %alloc_149, %alloc_139, %alloc_140, %alloc_145, %alloc_146, %alloc_142, %alloc_143, %alloc_129, %alloc_130, %alloc_125, %alloc_126, %alloc_122, %alloc_123, %alloc_119, %alloc_120, %alloc_115, %alloc_116, %alloc_106, %alloc_107, %alloc_112, %alloc_113, %alloc_109, %alloc_110, %alloc_96, %alloc_97, %alloc_92, %alloc_93, %alloc_89, %alloc_90, %alloc_86, %alloc_87, %alloc_82, %alloc_83, %alloc_79, %alloc_80, %alloc_76, %alloc_77, %alloc_154 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}
