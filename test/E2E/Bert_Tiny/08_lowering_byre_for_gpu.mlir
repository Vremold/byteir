// RUN: byteir-opt %s -func-tag="attach-attr=__placeholder__byre.entry_point func-name=main" -gen-ptx-config -convert-to-byre -byre-fold -cse | FileCheck %s

// CHECK-LABEL: func @main
module attributes {gpu.container_module}  {
  func private @MatmulOp0(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = memref.alloc() : memref<128x30522xf32>
    %1 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    "lmhlo.transpose"(%0, %1) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %1 : memref<30522x128xf32>
  }
  func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<2x128xi1>
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %2 = memref.alloc() : memref<2x128xi64>
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xui32>
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @Unknown0_kernel::@Unknown0_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128xi64>, %5 : memref<2x128xui32>, %2 : memref<2x128xi64>, %0 : memref<2x128xi1>)
    return %6, %4, %1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
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
  func private @Unknown1(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<2x128xi1>
    %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %2 = memref.alloc() : memref<2x128xi64>
    %3 = memref.collapse_shape %2 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %4 = memref.expand_shape %3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    %5 = memref.alloc() : memref<2x128xui32>
    %6 = memref.collapse_shape %5 [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    gpu.launch_func  @Unknown1_kernel::@Unknown1_kernel blocks in (%c8, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi64>, %5 : memref<2x128xui32>, %2 : memref<2x128xi64>, %0 : memref<2x128xi1>)
    return %6, %4, %1 : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
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
  func private @Unknown2(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %1 = memref.alloc() : memref<2x128x128xf32>
    %2 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    gpu.launch_func  @Unknown2_kernel::@Unknown2_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<2x128x128xf32>, %2 : memref<2x128x128xf32>, %1 : memref<2x128x128xf32>)
    return %1 : memref<2x128x128xf32>
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
  func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<128xi1>
    %1 = memref.alloc() : memref<128xi64>
    %2 = memref.expand_shape %1 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    %3 = memref.alloc() : memref<128xui32>
    %4 = memref.collapse_shape %arg0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    gpu.launch_func  @Unknown3_kernel::@Unknown3_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%4 : memref<128xi64>, %3 : memref<128xui32>, %1 : memref<128xi64>, %0 : memref<128xi1>)
    return %3, %2, %0 : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
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
  func private @Unknown4(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> memref<2x128x30522xf32> attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c244176 = arith.constant 244176 : index
    %c32 = arith.constant 32 : index
    %0 = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %1 = memref.alloc() : memref<2x128x30522xf32>
    gpu.launch_func  @Unknown4_kernel::@Unknown4_kernel blocks in (%c244176, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<2x128x30522xf32>, %arg1 : memref<30522xf32>, %1 : memref<2x128x30522xf32>)
    return %1 : memref<2x128x30522xf32>
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
  func private @Unknown5(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown5_kernel::@Unknown5_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown6(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    gpu.launch_func  @Unknown6_kernel::@Unknown6_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<2x128x128xf32>, %arg1 : memref<2x128x128xf32>, %arg2 : memref<2x128x128xf32>, %arg3 : memref<2x128x128xf32>, %0 : memref<2x128x128xf32>)
    return %0 : memref<2x128x128xf32>
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
  func private @Unknown7(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<2x128x128xf32>
    %1 = memref.collapse_shape %0 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %2 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %3 = memref.alloc() : memref<2x128x128xf32>
    %4 = memref.collapse_shape %3 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %5 = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    gpu.launch_func  @Unknown7_kernel::@Unknown7_kernel blocks in (%c1024, %c1, %c1) threads in (%c32, %c1, %c1) args(%5 : memref<2x128xi1>, %arg1 : memref<2x128x128xf32>, %3 : memref<2x128x128xf32>, %2 : memref<2x128xi1>, %0 : memref<2x128x128xf32>)
    return %4, %1 : memref<256x128xf32>, memref<256x128xf32>
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
  func private @Unknown8(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {byre_elementwise_fusion} {
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    %0 = memref.alloc() : memref<128x128xf32>
    gpu.launch_func  @Unknown8_kernel::@Unknown8_kernel blocks in (%c512, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128xi1>, %arg1 : memref<128x128xf32>, %0 : memref<128x128xf32>)
    return %0 : memref<128x128xf32>
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
  func @main(%arg0: memref<2x128xi64>, %arg1: memref<1x512xi64>, %arg2: memref<1x512xi64>, %arg3: memref<30522x128xf32>, %arg4: memref<2x128xf32>, %arg5: memref<512x128xf32>, %arg6: memref<128xf32>, %arg7: memref<128xf32>, %arg8: memref<128x128xf32>, %arg9: memref<128xf32>, %arg10: memref<128x128xf32>, %arg11: memref<128xf32>, %arg12: memref<128x128xf32>, %arg13: memref<128xf32>, %arg14: memref<2x1x1x128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<2x1x1x128xf32>, %arg32: memref<128x128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<128xf32>, %arg36: memref<512x128xf32>, %arg37: memref<512xf32>, %arg38: memref<128x512xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128xf32>, %arg42: memref<128x128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<128xf32>, %arg46: memref<30522xf32>, %arg47: memref<2x128x30522xf32>) -> (memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %0 = memref.alloc() : memref<2x128xf32>
    %1 = memref.alloc() : memref<30522xf32>
    %2 = memref.alloc() : memref<512x128xf32>
    %3 = memref.alloc() : memref<128x128xf32>
    %4 = memref.alloc() : memref<2x128xf32>
    %5 = memref.alloc() : memref<30522x128xf32>
    %6 = memref.alloc() : memref<2x128x128xf32>
    %7 = memref.alloc() : memref<128xf32>
    %8 = memref.alloc() : memref<128xf32>
    %9 = memref.alloc() : memref<2x128x128xf32>
    %10 = memref.alloc() : memref<128xf32>
    %11 = memref.alloc() : memref<128x128xf32>
    %12 = memref.alloc() : memref<2x128x128xf32>
    %13 = memref.alloc() : memref<128xf32>
    %14 = memref.alloc() : memref<128x128xf32>
    %15 = memref.alloc() : memref<2x128x128xf32>
    %16 = memref.alloc() : memref<128xf32>
    %17 = memref.alloc() : memref<128x128xf32>
    %18 = memref.alloc() : memref<2x128x128xf32>
    %19 = memref.alloc() : memref<2x2x128x64xf32>
    %20 = memref.alloc() : memref<2x2x128x64xf32>
    %21 = memref.alloc() : memref<2x2x128x128xf32>
    %22 = memref.alloc() : memref<2x2x128x64xf32>
    %23 = memref.alloc() : memref<2x2x128x128xf32>
    %24 = memref.alloc() : memref<2x2x128x64xf32>
    %25 = memref.alloc() : memref<2x128x2x64xf32>
    %26 = memref.alloc() : memref<128xf32>
    %27 = memref.alloc() : memref<128x128xf32>
    %28 = memref.alloc() : memref<2x128x128xf32>
    %29 = memref.alloc() : memref<2x128x128xf32>
    %30 = memref.alloc() : memref<128xf32>
    %31 = memref.alloc() : memref<128xf32>
    %32 = memref.alloc() : memref<2x128x128xf32>
    %33 = memref.alloc() : memref<2x128x128xf32>
    %34 = memref.alloc() : memref<512xf32>
    %35 = memref.alloc() : memref<512x128xf32>
    %36 = memref.alloc() : memref<2x128x128xf32>
    %37 = memref.alloc() : memref<128xf32>
    %38 = memref.alloc() : memref<128x512xf32>
    %39 = memref.alloc() : memref<2x128x512xf32>
    %40 = memref.alloc() : memref<2x128x128xf32>
    %41 = memref.alloc() : memref<128xf32>
    %42 = memref.alloc() : memref<128xf32>
    %43 = memref.alloc() : memref<2x128x128xf32>
    %44 = memref.alloc() : memref<128xf32>
    %45 = memref.alloc() : memref<128x128xf32>
    %46 = memref.alloc() : memref<2x128x128xf32>
    %47 = memref.alloc() : memref<128xf32>
    %48 = memref.alloc() : memref<128x128xf32>
    %49 = memref.alloc() : memref<2x128x128xf32>
    %50 = memref.alloc() : memref<128xf32>
    %51 = memref.alloc() : memref<128x128xf32>
    %52 = memref.alloc() : memref<2x128x128xf32>
    %53 = memref.alloc() : memref<2x2x128x64xf32>
    %54 = memref.alloc() : memref<2x2x128x64xf32>
    %55 = memref.alloc() : memref<2x2x128x128xf32>
    %56 = memref.alloc() : memref<2x2x128x64xf32>
    %57 = memref.alloc() : memref<2x2x128x128xf32>
    %58 = memref.alloc() : memref<2x2x128x64xf32>
    %59 = memref.alloc() : memref<2x128x2x64xf32>
    %60 = memref.alloc() : memref<128xf32>
    %61 = memref.alloc() : memref<128x128xf32>
    %62 = memref.alloc() : memref<2x128x128xf32>
    %63 = memref.alloc() : memref<2x128x128xf32>
    %64 = memref.alloc() : memref<128xf32>
    %65 = memref.alloc() : memref<128xf32>
    %66 = memref.alloc() : memref<2x128x128xf32>
    %67 = memref.alloc() : memref<2x128x128xf32>
    %68 = memref.alloc() : memref<512xf32>
    %69 = memref.alloc() : memref<512x128xf32>
    %70 = memref.alloc() : memref<2x128x128xf32>
    %71 = memref.alloc() : memref<128xf32>
    %72 = memref.alloc() : memref<128x512xf32>
    %73 = memref.alloc() : memref<2x128x512xf32>
    %74 = memref.alloc() : memref<2x128x128xf32>
    %75 = memref.alloc() : memref<128xf32>
    %76 = memref.alloc() : memref<128xf32>
    %77 = memref.alloc() : memref<2x128x128xf32>
    %78 = memref.alloc() : memref<128xf32>
    %79 = memref.alloc() : memref<128x128xf32>
    %80 = memref.alloc() : memref<2x128x128xf32>
    %81 = memref.alloc() : memref<128xf32>
    %82 = memref.alloc() : memref<128xf32>
    %83 = memref.alloc() : memref<2x128x128xf32>
    %84 = memref.alloc() : memref<256x30522xf32>
    %85 = memref.alloc() : memref<256x128xf32>
    %86 = memref.alloc() : memref<256xf32>
    %87 = memref.alloc() : memref<256xf32>
    %88 = memref.alloc() : memref<2x128x128xf32>
    %89 = memref.alloc() : memref<0xf32>
    %90 = memref.alloc() : memref<2x128x128xf32>
    %91 = memref.alloc() : memref<2x128x128xf32>
    %92 = memref.alloc() : memref<2x128x128xf32>
    %93 = memref.alloc() : memref<256xf32>
    %94 = memref.alloc() : memref<256xf32>
    %95 = memref.alloc() : memref<2x128x128xf32>
    %96 = memref.alloc() : memref<2x128x128xui8>
    %97 = memref.alloc() : memref<2x128x128xf32>
    %98 = memref.alloc() : memref<2x128x128xf32>
    %99 = memref.alloc() : memref<0xf32>
    %100 = memref.alloc() : memref<2x128x512xf32>
    %101 = memref.alloc() : memref<2x128x512xf32>
    %102 = memref.alloc() : memref<2x128x128xf32>
    %103 = memref.alloc() : memref<256xf32>
    %104 = memref.alloc() : memref<256xf32>
    %105 = memref.alloc() : memref<2x128x128xf32>
    %106 = memref.alloc() : memref<2x128x128xui8>
    %107 = memref.alloc() : memref<2x128x128xf32>
    %108 = memref.alloc() : memref<2x128x128xf32>
    %109 = memref.alloc() : memref<2x128x128xf32>
    %110 = memref.alloc() : memref<2x128x2x64xf32>
    %111 = memref.alloc() : memref<2x2x128x64xf32>
    %112 = memref.alloc() : memref<2x2x128x64xf32>
    %113 = memref.alloc() : memref<2x2x128x128xui8>
    %114 = memref.alloc() : memref<2x2x128x128xf32>
    %115 = memref.alloc() : memref<2x2x128x128xf32>
    %116 = memref.alloc() : memref<2x2x128x128xf32>
    %117 = memref.alloc() : memref<2x2x128x64xf32>
    %118 = memref.alloc() : memref<2x2x128x64xf32>
    %119 = memref.alloc() : memref<2x128x128xf32>
    %120 = memref.alloc() : memref<256xf32>
    %121 = memref.alloc() : memref<256xf32>
    %122 = memref.alloc() : memref<2x128x128xf32>
    %123 = memref.alloc() : memref<2x128x128xui8>
    %124 = memref.alloc() : memref<2x128x128xf32>
    %125 = memref.alloc() : memref<2x128x128xf32>
    %126 = memref.alloc() : memref<0xf32>
    %127 = memref.alloc() : memref<2x128x512xf32>
    %128 = memref.alloc() : memref<2x128x512xf32>
    %129 = memref.alloc() : memref<2x128x128xf32>
    %130 = memref.alloc() : memref<256xf32>
    %131 = memref.alloc() : memref<256xf32>
    %132 = memref.alloc() : memref<2x128x128xf32>
    %133 = memref.alloc() : memref<2x128x128xui8>
    %134 = memref.alloc() : memref<2x128x128xf32>
    %135 = memref.alloc() : memref<2x128x128xf32>
    %136 = memref.alloc() : memref<2x128x128xf32>
    %137 = memref.alloc() : memref<2x128x2x64xf32>
    %138 = memref.alloc() : memref<2x2x128x64xf32>
    %139 = memref.alloc() : memref<2x2x128x64xf32>
    %140 = memref.alloc() : memref<2x2x128x128xui8>
    %141 = memref.alloc() : memref<2x2x128x128xf32>
    %142 = memref.alloc() : memref<2x2x128x128xf32>
    %143 = memref.alloc() : memref<2x2x128x128xf32>
    %144 = memref.alloc() : memref<2x2x128x64xf32>
    %145 = memref.alloc() : memref<2x2x128x64xf32>
    %146 = memref.alloc() : memref<2x128x128xf32>
    %147 = memref.alloc() : memref<256xf32>
    %148 = memref.alloc() : memref<256xf32>
    %149 = memref.alloc() : memref<2x128x128xf32>
    %150 = memref.alloc() : memref<1x128x128xf32>
    %151 = memref.alloc() : memref<128x128xf32>
    %152 = memref.alloc() : memref<256x128xf32>
    %153 = memref.alloc() : memref<2x128x128xf32>
    %154 = memref.alloc() : memref<256x128xf32>
    %155 = memref.alloc() : memref<256x128xf32>
    %156 = memref.alloc() : memref<256x30522xf32>
    %157 = memref.alloc() : memref<1x128xi64>
    %158 = memref.alloc() : memref<128xi64>
    %159 = memref.alloc() : memref<1x128xi64>
    %160 = memref.alloc() : memref<f32>
    %161 = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    "lmhlo.constant"(%161) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    "lmhlo.constant"(%160) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    "lmhlo.slice"(%arg1, %159) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%159, %158) : (memref<1x128xi64>, memref<128xi64>) -> ()
    "lmhlo.slice"(%arg2, %157) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.reshape"(%arg47, %156) : (memref<2x128x30522xf32>, memref<256x30522xf32>) -> ()
    %162:3 = call @Unknown0(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg3, %162#0, %155) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%156, %arg3, %154) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.reshape"(%154, %153) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %163:3 = call @Unknown1(%158) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    "lmhlo.gather"(%arg4, %163#0, %152) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %164 = call @Unknown2(%155, %152) : (memref<256x128xf32>, memref<256x128xf32>) -> memref<2x128x128xf32>
    %165:3 = call @Unknown3(%157) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    "lmhlo.gather"(%arg5, %165#0, %151) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    "lmhlo.reshape"(%151, %150) : (memref<128x128xf32>, memref<1x128x128xf32>) -> ()
    "lmhlo.custom_call"(%164, %arg6, %arg7, %150, %149, %148, %147, %146) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<1x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%149, %arg8, %arg9, %145) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%149, %arg10, %arg11, %144) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%145, %144, %143) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%143, %arg14, %142, %141, %140) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%149, %arg12, %arg13, %139) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%141, %139, %138) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%138, %137) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%137, %136) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%136, %arg15, %arg16, %135, %134, %133) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%135, %arg17, %arg18, %149, %132, %131, %130, %129) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%132, %arg19, %arg20, %128, %127, %126) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%128, %arg21, %arg22, %125, %124, %123) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%125, %arg23, %arg24, %132, %122, %121, %120, %119) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%122, %arg25, %arg26, %118) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%122, %arg27, %arg28, %117) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%118, %117, %116) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%116, %arg31, %115, %114, %113) {api_version = 1 : i32, backend_config = "{batch_first = true, dropout_rate = 1.000000e-01 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", has_side_effect = false, operand_segment_sizes = dense<[2, 3]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x1x1x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%122, %arg29, %arg30, %112) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%114, %112, %111) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%111, %110) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.reshape"(%110, %109) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%109, %arg32, %arg33, %108, %107, %106) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%108, %arg34, %arg35, %122, %105, %104, %103, %102) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%105, %arg36, %arg37, %101, %100, %99) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%101, %arg38, %arg39, %98, %97, %96) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>) -> ()
    "lmhlo.custom_call"(%98, %arg40, %arg41, %105, %95, %94, %93, %92) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_residual", has_side_effect = false, operand_segment_sizes = dense<4> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%95, %arg42, %arg43, %91, %90, %89) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    "lmhlo.custom_call"(%91, %arg44, %arg45, %88, %87, %86) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    "lmhlo.reshape"(%88, %85) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    "lmhlo.dot"(%85, %arg3, %84) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = ["DEFAULT", "DEFAULT"]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %166 = call @Unknown4(%84, %arg46) : (memref<256x30522xf32>, memref<30522xf32>) -> memref<2x128x30522xf32>
    %167 = call @MatmulOp0(%85, %156) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    "lmhlo.custom_call"(%153, %91, %arg44, %87, %86, %83, %82, %81) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%83, %95, %arg42, %90, %89, %80, %79, %78) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%80, %92, %arg40, %94, %93, %77, %76, %75, %74) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%77, %101, %arg38, %97, %96, %73, %72, %71) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%73, %105, %arg36, %100, %99, %70, %69, %68) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%74, %70, %67) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%67, %102, %arg34, %104, %103, %66, %65, %64, %63) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%66, %109, %arg32, %107, %106, %62, %61, %60) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%62, %59) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%59, %58) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%58, %114, %112, %57, %56) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%57, %115, %113, %55) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%55, %118, %117, %54, %53) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%54, %122, %arg25, %52, %51, %50) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%56, %122, %arg29, %49, %48, %47) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%53, %122, %arg27, %46, %45, %44) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %168 = call @Unknown5(%63, %52, %49, %46) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%168, %119, %arg23, %121, %120, %43, %42, %41, %40) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%43, %128, %arg21, %124, %123, %39, %38, %37) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%39, %132, %arg19, %127, %126, %36, %35, %34) {api_version = 1 : i32, backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    "lmhlo.add"(%40, %36, %33) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%33, %129, %arg17, %131, %130, %32, %31, %30, %29) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%32, %136, %arg15, %134, %133, %28, %27, %26) {api_version = 1 : i32, backend_config = "{act_gelu = false, dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", has_side_effect = false, operand_segment_sizes = dense<[5, 3]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xui8>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.reshape"(%28, %25) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    "lmhlo.custom_call"(%25, %24) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", has_side_effect = false, operand_segment_sizes = dense<1> : vector<2xi32>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%24, %141, %139, %23, %22) {api_version = 1 : i32, backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%23, %142, %140, %21) {api_version = 1 : i32, backend_config = "{dropout_rate = 1.000000e-01 : f32}", call_target_name = "ftv4.softmax_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 1]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    "lmhlo.custom_call"(%21, %145, %144, %20, %19) {api_version = 1 : i32, backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", has_side_effect = false, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    "lmhlo.custom_call"(%20, %149, %arg8, %18, %17, %16) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%22, %149, %arg12, %15, %14, %13) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    "lmhlo.custom_call"(%19, %149, %arg10, %12, %11, %10) {api_version = 1 : i32, backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", has_side_effect = false, operand_segment_sizes = dense<3> : vector<2xi32>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %169 = call @Unknown6(%29, %18, %15, %12) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    "lmhlo.custom_call"(%169, %146, %arg6, %148, %147, %9, %8, %7, %6) {api_version = 1 : i32, backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", has_side_effect = false, operand_segment_sizes = dense<[5, 4]> : vector<2xi32>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %170:2 = call @Unknown7(%162#2, %9, %163#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    "lmhlo.scatter"(%167, %162#1, %170#0, %5) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    "lmhlo.scatter"(%0, %163#1, %170#1, %4) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    "lmhlo.reduce"(%6, %160, %3) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %171 = call @Unknown8(%165#2, %3) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    "lmhlo.scatter"(%161, %165#1, %171, %2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %172 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%172) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    "lmhlo.reduce"(%arg47, %160, %1) ( {
    ^bb0(%arg48: memref<f32>, %arg49: memref<f32>, %arg50: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg48, %arg49, %arg50) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %166, %5, %4, %2, %8, %7, %17, %16, %11, %10, %14, %13, %27, %26, %31, %30, %35, %34, %38, %37, %42, %41, %51, %50, %45, %44, %48, %47, %61, %60, %65, %64, %69, %68, %72, %71, %76, %75, %79, %78, %82, %81, %1 : memref<2x128x30522xf32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

